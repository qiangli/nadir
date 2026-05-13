package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/qiangli/nadir/provider/openai"
	"github.com/qiangli/nadir/skillrouter"
)

// ollamaBaseURL resolves the base URL the eval uses to reach Ollama.
// NADIR_OLLAMA_BASE_URL overrides; otherwise the standard local
// endpoint. Returns "" if the caller explicitly disabled Ollama via
// NADIR_OLLAMA_BASE_URL=disabled.
func ollamaBaseURL() string {
	v := os.Getenv("NADIR_OLLAMA_BASE_URL")
	if v == "disabled" {
		return ""
	}
	if v != "" {
		return v
	}
	return "http://localhost:11434"
}

// ollamaCascadeLLMModel is the small text-completion model used for
// the LLM rerank in the Cascade matcher row. The user can override
// with NADIR_CASCADE_LLM_MODEL.
func ollamaCascadeLLMModel() string {
	if v := os.Getenv("NADIR_CASCADE_LLM_MODEL"); v != "" {
		return v
	}
	return "qwen3.5:0.8b"
}

// ollamaEmbeddingModels lists Ollama models to probe as drop-in MiniLM
// replacements. Curated for the "small embedders that beat MiniLM"
// question — biased toward minimalist (small param count) candidates,
// with one larger model included as a ceiling case. MTEB scores are
// approximate (mteb leaderboard 2026):
//
//	all-minilm:l6-v2                ~33M  / 384d   / ~56 MTEB (the reference baseline via Ollama)
//	snowflake-arctic-embed:33m       33M  / 384d   / ~60 MTEB (small competitor)
//	nomic-embed-text:latest         137M  / 768d   / ~62 MTEB
//	mxbai-embed-large:latest        335M  / 1024d  / ~65 MTEB
//	qwen3-embedding:8b               7.6B / 4096d  / ~71 MTEB (ceiling)
//
// Probes are best-effort: models not pulled / Ollama unreachable are
// reported to stderr and skipped.
var ollamaEmbeddingModels = []string{
	"all-minilm:latest",
	"snowflake-arctic-embed:33m",
	"nomic-embed-text:latest",
	"mxbai-embed-large:latest",
	"qwen3-embedding:8b",
}

// tryOllamaEmbedders appends Ollama-backed Embedder rows for each
// model that the running Ollama instance has pulled. Each probe takes
// one HTTP round-trip; total construction cost is bounded by
// len(models) × cold-load latency.
func tryOllamaEmbedders(ctx context.Context, baseURL string) []namedEmbedder {
	if baseURL == "" {
		return nil
	}
	var out []namedEmbedder
	for _, m := range ollamaEmbeddingModels {
		probeCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
		emb, err := skillrouter.NewOllamaEmbedder(probeCtx, baseURL, m,
			skillrouter.WithOllamaTimeout(60*time.Second),
		)
		cancel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "ollama embedder %q unavailable: %v\n", m, err)
			continue
		}
		out = append(out, namedEmbedder{
			name: fmt.Sprintf("Ollama: %s (%dd)", m, emb.Dim()),
			emb:  emb,
		})
	}
	return out
}

// tryOllamaCascadeMatcher builds the Cascade(TF-IDF primary + Ollama
// LLM rerank on shortlist) matcher. The pure-Go TF-IDF handles the
// fast path; the LLM is consulted only when primary's margin is
// narrow OR the prompt looks OOD-ish. Returns ok=false if Ollama
// isn't reachable.
func tryOllamaCascadeMatcher(ctx context.Context, baseURL, llmModel string, skills []skillrouter.Skill) (namedMatcher, bool) {
	if baseURL == "" || llmModel == "" {
		return namedMatcher{}, false
	}
	primary, err := skillrouter.NewSemantic(ctx, skillrouter.NewTFIDFFromSkills(skills), skills,
		skillrouter.WithMinScore(0.20),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cascade primary build failed: %v\n", err)
		return namedMatcher{}, false
	}
	llm := openai.New("ollama", baseURL+"/v1", "")
	cas := skillrouter.NewCascade(primary, llm, llmModel,
		skillrouter.WithMinMargin(0.10),
		skillrouter.WithShortlistK(3),
		skillrouter.WithLLMTimeout(20*time.Second),
	)
	return namedMatcher{
		name:    fmt.Sprintf("Cascade: TF-IDF + Ollama %s rerank", llmModel),
		matcher: cas,
	}, true
}
