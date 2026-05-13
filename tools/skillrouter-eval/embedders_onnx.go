//go:build onnx

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/qiangli/nadir/embed"
	"github.com/qiangli/nadir/skillrouter"
)

// buildEmbedders returns the pure-Go embedders, then MiniLM-L6-v2 via
// ONNX when assets/ is reachable, then any reachable Ollama embedding
// models. MiniLM failure is logged but not fatal — the eval still
// produces a useful comparison of the remaining options.
func buildEmbedders(assetsPath string, skills []skillrouter.Skill) []namedEmbedder {
	out := []namedEmbedder{
		{name: "TF-IDF (pure Go)", emb: skillrouter.NewTFIDFFromSkills(skills)},
		{name: "Hashing 256d 3-5gram (pure Go)", emb: skillrouter.NewHashEmbedder()},
	}
	if ens, err := skillrouter.NewLexicalEnsemble(skills); err == nil {
		out = append(out, namedEmbedder{name: "Ensemble: TF-IDF + Hashing (pure Go)", emb: ens})
	}
	if mini, err := embed.Open(assetsPath); err != nil {
		fmt.Fprintf(os.Stderr, "MiniLM unavailable (continuing with pure-Go embedders only): %v\n", err)
	} else {
		out = append(out, namedEmbedder{name: "MiniLM-L6-v2 (ONNX, 384d)", emb: mini})
	}
	out = append(out, tryOllamaEmbedders(context.Background(), ollamaBaseURL())...)
	return out
}
