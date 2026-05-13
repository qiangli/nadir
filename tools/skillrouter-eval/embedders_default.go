//go:build !onnx

package main

import (
	"context"

	"github.com/qiangli/nadir/skillrouter"
)

// buildEmbedders returns the pure-Go embedders plus any reachable
// Ollama embedding models. The MiniLM (ONNX) row is contributed by
// the `onnx` build tag.
func buildEmbedders(_ string, skills []skillrouter.Skill) []namedEmbedder {
	out := []namedEmbedder{
		{name: "TF-IDF (pure Go)", emb: skillrouter.NewTFIDFFromSkills(skills)},
		{name: "Hashing 256d 3-5gram (pure Go)", emb: skillrouter.NewHashEmbedder()},
	}
	if ens, err := skillrouter.NewLexicalEnsemble(skills); err == nil {
		out = append(out, namedEmbedder{name: "Ensemble: TF-IDF + Hashing (pure Go)", emb: ens})
	}
	out = append(out, tryOllamaEmbedders(context.Background(), ollamaBaseURL())...)
	return out
}
