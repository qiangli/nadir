package skillrouter

import (
	"context"
	"errors"
	"fmt"
)

// Ensemble is an Embedder that combines two underlying embedders by
// concatenating their (optionally re-weighted) output vectors and
// L2-normalizing the result. The intent: pair complementary signals —
// e.g., TFIDF (vocabulary-level, captures word identity) with
// HashEmbedder (subword-level, captures morphology and typos) —
// so the Semantic matcher sees both.
//
// Why it works: TFIDF and HashEmbedder have partly independent error
// modes. On the skill-router eval corpus, TFIDF wins paraphrase cases
// (92.9% vs 78.6%) and HashEmbedder wins overlap cases (100% vs 80%).
// Concatenating in the dot-product space lets a skill that's well-
// covered by either signal score high — without letting OOD prompts
// pile up signal from either side.
//
// Numerics: each underlying vector is L2-normalized (the convention
// nadir's Embedder interface guarantees). Concatenation gives a
// (dim_a + dim_b)-length vector with norm √(α² + β²); re-normalizing
// at the end keeps cosine in [-1, 1]. Equivalently, the cosine in the
// combined space is a weighted average of the two source cosines:
//
//	cos(C(p), C(s)) = (α²·cos_a(p,s) + β²·cos_b(p,s)) / (α² + β²)
//
// With Weights = [1, 1] (the default), the combined cosine is exactly
// the average of the two — the simplest sane fusion.
type Ensemble struct {
	a, b      Embedder
	weightA   float32
	weightB   float32
	combinedD int
}

// EnsembleOption configures an Ensemble.
type EnsembleOption func(*Ensemble)

// WithEnsembleWeights sets the per-embedder scalar weights. Internally
// they multiply each vector before concatenation, so weight=2 doubles
// the contribution of that side. Default is [1, 1] (equal weight).
// Pass zero to a side to disable it (degenerate but supported).
func WithEnsembleWeights(a, b float32) EnsembleOption {
	return func(e *Ensemble) {
		if a >= 0 {
			e.weightA = a
		}
		if b >= 0 {
			e.weightB = b
		}
	}
}

// NewEnsemble wires two embedders into a combined one. Returns an
// error if either is nil. The combined Dim() is a.Dim() + b.Dim().
func NewEnsemble(a, b Embedder, opts ...EnsembleOption) (*Ensemble, error) {
	if a == nil || b == nil {
		return nil, errors.New("skillrouter: nil embedder in ensemble")
	}
	e := &Ensemble{
		a:         a,
		b:         b,
		weightA:   1,
		weightB:   1,
		combinedD: a.Dim() + b.Dim(),
	}
	for _, opt := range opts {
		opt(e)
	}
	return e, nil
}

// Embed runs both underlying embedders, concatenates with weights, and
// L2-normalizes. Returns the underlying error if either side fails —
// no silent partial output.
func (e *Ensemble) Embed(ctx context.Context, text string) ([]float32, error) {
	va, err := e.a.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("ensemble: a.Embed: %w", err)
	}
	vb, err := e.b.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("ensemble: b.Embed: %w", err)
	}
	if len(va) != e.a.Dim() || len(vb) != e.b.Dim() {
		return nil, errors.New("ensemble: embedder dim mismatch with declared Dim()")
	}
	out := make([]float32, e.combinedD)
	for i, v := range va {
		out[i] = v * e.weightA
	}
	for i, v := range vb {
		out[len(va)+i] = v * e.weightB
	}
	l2Normalize(out)
	return out, nil
}

// Dim returns the concatenated dimension.
func (e *Ensemble) Dim() int { return e.combinedD }

// Close closes both underlying embedders, returning the first error
// encountered. A closed Ensemble cannot be reused.
func (e *Ensemble) Close() error {
	errA := e.a.Close()
	errB := e.b.Close()
	if errA != nil {
		return errA
	}
	return errB
}

// NewLexicalEnsemble is the recommended "best pure-Go default":
// TF-IDF + Hashing 3-5gram, equal weight. Vocabulary is fit from the
// skill catalog's own exemplars. Useful one-liner for the common case;
// for finer control build the components yourself and pass to
// NewEnsemble directly.
func NewLexicalEnsemble(skills []Skill) (*Ensemble, error) {
	return NewEnsemble(NewTFIDFFromSkills(skills), NewHashEmbedder())
}

var _ Embedder = (*Ensemble)(nil)
