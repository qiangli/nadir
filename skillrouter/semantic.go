package skillrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sort"
)

// Embedder is the seam between this package and the embed package.
// Kept local (mirrors classifier.Embedder) so the Semantic matcher
// doesn't transitively import embed in tests where we substitute a
// fake. Vectors are expected to be L2-normalized so dot product is
// cosine similarity — the convention nadir's embed.Open guarantees.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Dim() int
	Close() error
}

// Semantic picks a skill by embedding the prompt and scoring it
// against per-skill multi-vector prototypes. Score for a skill is the
// max cosine over its prototype vectors — the multi-vector trick from
// the CLINC150 literature (~15% lift over single-centroid retrieval).
//
// Two signals come out: top-1 cosine (Confidence) and top1-top2 gap
// (Margin). Cascade uses Margin to decide whether to consult an LLM
// secondary, because a top-1 of 0.78 with a runner-up of 0.77 is
// uncertain even though 0.78 looks high.
//
// FellThrough fires when top-1 falls below MinScore — the embedding
// space says "nothing in the catalog is close." Callers should bypass
// skill dispatch in that case.
type Semantic struct {
	emb       Embedder
	skills    []Skill
	protoVecs [][][]float32 // protoVecs[skillIdx] = [][]float32, one vec per exemplar
	minScore  float64
	logger    *slog.Logger
}

// SemanticOption configures the Semantic matcher.
type SemanticOption func(*Semantic)

// WithMinScore sets the top-1 cosine threshold below which Route
// returns FellThrough=true. Default 0.35 — calibrated for L2-normalized
// MiniLM embeddings where unrelated short prompts cluster around
// 0.1–0.25. Tune higher (~0.5) if your skill catalog is very specific.
func WithMinScore(s float64) SemanticOption {
	return func(m *Semantic) { m.minScore = s }
}

// WithSemanticLogger sets the slog.Logger used for warning-level
// diagnostics. Defaults to slog.Default().
func WithSemanticLogger(l *slog.Logger) SemanticOption {
	return func(m *Semantic) {
		if l != nil {
			m.logger = l
		}
	}
}

// NewSemantic builds a Semantic matcher and embeds every exemplar in
// the catalog up front. The embedding work is one-time at startup;
// Route is then a single Embed call plus N*M cosine dot products.
//
// Exemplar selection per skill (highest-quality source wins):
//  1. Examples — embedded one vector per entry (multi-vector prototype)
//  2. Description — embedded as a single fallback vector
//  3. Name — last-resort vector so the matcher still functions
//
// Returns an error if the catalog is empty or every skill is unusable
// (no Name and no exemplar text).
func NewSemantic(ctx context.Context, emb Embedder, skills []Skill, opts ...SemanticOption) (*Semantic, error) {
	if emb == nil {
		return nil, errors.New("skillrouter: nil embedder")
	}
	if len(skills) == 0 {
		return nil, errors.New("skillrouter: empty skill catalog")
	}

	s := &Semantic{
		emb:      emb,
		skills:   skills,
		minScore: 0.35,
		logger:   slog.Default(),
	}
	for _, opt := range opts {
		opt(s)
	}

	s.protoVecs = make([][][]float32, len(skills))
	for i, sk := range skills {
		texts := exemplarTexts(sk)
		if len(texts) == 0 {
			return nil, fmt.Errorf("skillrouter: skill %q has no Name/Description/Examples", sk.Name)
		}
		vecs := make([][]float32, 0, len(texts))
		for _, t := range texts {
			v, err := emb.Embed(ctx, t)
			if err != nil {
				return nil, fmt.Errorf("skillrouter: embed %q for skill %q: %w", t, sk.Name, err)
			}
			vecs = append(vecs, v)
		}
		s.protoVecs[i] = vecs
	}
	return s, nil
}

// exemplarTexts picks the strings we embed for a skill, in preference
// order: Examples beats Description beats Name. We never silently
// pull from multiple sources — a skill that ships Examples but a
// noisy Description shouldn't have the Description bleed into the
// prototype space.
func exemplarTexts(s Skill) []string {
	if len(s.Examples) > 0 {
		out := make([]string, 0, len(s.Examples))
		for _, e := range s.Examples {
			if e != "" {
				out = append(out, e)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	if s.Description != "" {
		return []string{s.Description}
	}
	if s.Name != "" {
		return []string{s.Name}
	}
	return nil
}

// Route picks the best-scoring skill or returns FellThrough=true when
// the top score is below MinScore.
func (s *Semantic) Route(ctx context.Context, prompt string) (*Decision, error) {
	ranked, err := s.Rank(ctx, prompt, 0)
	if err != nil {
		return nil, err
	}
	if len(ranked) == 0 {
		return &Decision{FellThrough: true}, nil
	}
	top := ranked[0]
	margin := 0.0
	if len(ranked) > 1 {
		margin = top.Score - ranked[1].Score
	}
	if top.Score < s.minScore {
		s.logger.Debug("semantic: fellthrough",
			slog.Float64("top_score", top.Score),
			slog.Float64("min_score", s.minScore),
		)
		return &Decision{FellThrough: true, Confidence: top.Score, Margin: margin}, nil
	}
	return &Decision{Skill: top.Skill, Confidence: top.Score, Margin: margin}, nil
}

// Ranked is one entry in a Rank() result.
type Ranked struct {
	Skill string
	Score float64
}

// Rank embeds the prompt once and returns the catalog sorted by
// descending max-cosine score. Pass k=0 to get every skill ranked;
// k>0 truncates to the top-k.
//
// Cascade uses this to build a shortlist for its LLM secondary — the
// RAG-style "retrieve, then rerank" pattern.
func (s *Semantic) Rank(ctx context.Context, prompt string, k int) ([]Ranked, error) {
	if prompt == "" {
		return nil, errors.New("skillrouter: empty prompt")
	}
	qv, err := s.emb.Embed(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("skillrouter: embed prompt: %w", err)
	}
	out := make([]Ranked, len(s.skills))
	for i, vecs := range s.protoVecs {
		best := math.Inf(-1)
		for _, v := range vecs {
			c := cosine(qv, v)
			if c > best {
				best = c
			}
		}
		out[i] = Ranked{Skill: s.skills[i].Name, Score: best}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if k > 0 && k < len(out) {
		out = out[:k]
	}
	return out, nil
}

// Close releases the underlying embedder. Safe to call once; idempotent
// only if the embedder itself is.
func (s *Semantic) Close() error {
	if s.emb != nil {
		return s.emb.Close()
	}
	return nil
}

// cosine assumes both vectors are L2-normalized (the embed package's
// guarantee) so we skip the magnitude divisions and reduce to a dot
// product. If a caller plugs in a non-normalized embedder, scores
// drift but ordering is preserved — Cascade's margin signal still
// works.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var sum float64
	for i, v := range a {
		sum += float64(v) * float64(b[i])
	}
	return sum
}

var _ Matcher = (*Semantic)(nil)
