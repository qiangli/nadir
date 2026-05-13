package skillrouter

import (
	"context"
	"errors"
	"time"

	"github.com/qiangli/nadir/types"
)

// NewLexical is the no-dependencies production default: a Semantic
// matcher backed by a TF-IDF embedder fitted from the catalog's own
// exemplars, with MinScore calibrated for TF-IDF's score distribution
// (an OOD prompt averages ~0.20 here; a confident in-catalog hit
// clears 0.40).
//
// Quality on the bundled eval corpus: ~94% top-1 accuracy, ~50% OOD
// rejection. Pure Go, ~1.6 µs per Route call.
//
// Use this when:
//   - The binary must be static — no Ollama, no CGO
//   - Skills ship rich Examples (3+ paraphrases each)
//   - Wrong-routes are recoverable (a default handler exists)
//
// For OOD-sensitive deployments, prefer NewHybrid — the LLM rerank
// converts narrow-margin and unfamiliar prompts into clean
// FellThrough signals at the cost of an Ollama dependency.
//
// For finer control (custom MinScore, alternative embedders), build
// the matcher directly from NewSemantic.
func NewLexical(skills []Skill) (*Semantic, error) {
	if len(skills) == 0 {
		return nil, errors.New("skillrouter: empty skill catalog")
	}
	return NewSemantic(context.Background(), NewTFIDFFromSkills(skills), skills,
		WithMinScore(0.25),
	)
}

// NewHybrid is the recommended production matcher when Ollama is
// available: TF-IDF primary handles the easy ~80% of prompts in
// microseconds, the LLM is consulted only when the primary's top-1 is
// uncertain (cosine below MinScore=0.25 or top1-top2 margin below
// 0.10). Silent fallback to the primary on LLM error means routing
// never returns a 5xx — the worst case is one sub-optimal route.
//
// Quality on the bundled eval corpus with qwen2.5:7b as the rerank
// model: 97.1% top-1 accuracy, ~83% OOD rejection. Hot path stays at
// ~2 µs; LLM tail latency is ~300–400 ms only on the uncertain ~20%.
//
// llmClient should point at Ollama (or any OpenAI-compatible
// endpoint). llmModel is the rerank model — qwen2.5:7b in the eval,
// but any small instruction-tuned model works; quality scales with
// the model. For very tight latency budgets, a 1–3B model (e.g.,
// llama3.2:1b) trades accuracy for speed.
//
// For finer control (custom MinScore/Margin/timeout, alternative
// primary embedders, custom logger), build the matcher directly:
//
//	primary, _ := NewSemantic(ctx, NewTFIDFFromSkills(skills), skills, ...)
//	cas := NewCascade(primary, llmClient, llmModel, ...)
func NewHybrid(ctx context.Context, llmClient types.LLMClient, llmModel string, skills []Skill) (*Cascade, error) {
	if len(skills) == 0 {
		return nil, errors.New("skillrouter: empty skill catalog")
	}
	if llmClient == nil {
		return nil, errors.New("skillrouter: nil LLM client")
	}
	if llmModel == "" {
		return nil, errors.New("skillrouter: empty LLM model")
	}
	primary, err := NewSemantic(ctx, NewTFIDFFromSkills(skills), skills,
		WithMinScore(0.25),
	)
	if err != nil {
		return nil, err
	}
	return NewCascade(primary, llmClient, llmModel,
		WithMinMargin(0.10),
		WithShortlistK(5),
		WithLLMTimeout(2*time.Second),
	), nil
}
