package skillrouter

import (
	"context"
	"errors"
	"math"
	"testing"
)

// fakeEmbedder maps known strings to canned unit vectors. Unmapped
// strings get a zero vector (low cosine vs anything). dim is whatever
// the test set up.
type fakeEmbedder struct {
	vectors map[string][]float32
	dim     int
	err     error
}

func (f *fakeEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	if v, ok := f.vectors[text]; ok {
		out := make([]float32, len(v))
		copy(out, v)
		return out, nil
	}
	return make([]float32, f.dim), nil
}

func (f *fakeEmbedder) Dim() int     { return f.dim }
func (f *fakeEmbedder) Close() error { return nil }

// unit returns an L2-normalized vector. Test data is hand-built so
// cosine scores are predictable.
func unit(v ...float32) []float32 {
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	norm := float32(math.Sqrt(sumSq))
	if norm == 0 {
		return v
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// 4D skill space: skills occupy axes 0-2; axis 3 is reserved for
// "unrelated" prompts so we can construct a vector that's far from
// every catalog exemplar without anti-aligning anything.
func newFakeEmb() *fakeEmbedder {
	return &fakeEmbedder{
		dim: 4,
		vectors: map[string][]float32{
			// Review exemplars cluster on axis 0.
			"review pull request": unit(1, 0, 0, 0),
			"look at this diff":   unit(0.95, 0.1, 0, 0),
			"review the changes":  unit(0.9, 0, 0.1, 0),
			// Security exemplars cluster on axis 1.
			"audit for vulnerabilities": unit(0, 1, 0, 0),
			"security check":            unit(0.1, 0.95, 0, 0),
			// Init exemplars cluster on axis 2.
			"initialize CLAUDE.md": unit(0, 0, 1, 0),
			// Test prompts.
			"please review my PR":       unit(0.95, 0.05, 0, 0),
			"check this for security":   unit(0.1, 0.9, 0.1, 0),
			"close-call: review or sec": unit(0.7, 0.65, 0, 0), // narrow margin
			// Lives entirely on the spare axis — cosine ~0 vs every exemplar.
			"unrelated random text": unit(0, 0, 0, 1),
		},
	}
}

func newTestCatalog() []Skill {
	return []Skill{
		{
			Name: "/review",
			Examples: []string{
				"review pull request",
				"look at this diff",
				"review the changes",
			},
		},
		{
			Name: "/security-review",
			Examples: []string{
				"audit for vulnerabilities",
				"security check",
			},
		},
		{
			Name: "/init",
			Examples: []string{
				"initialize CLAUDE.md",
			},
		},
	}
}

func TestSemantic_PicksBestSkill(t *testing.T) {
	emb := newFakeEmb()
	s, err := NewSemantic(context.Background(), emb, newTestCatalog())
	if err != nil {
		t.Fatal(err)
	}
	d, err := s.Route(context.Background(), "please review my PR")
	if err != nil {
		t.Fatal(err)
	}
	if d.Skill != "/review" || d.FellThrough {
		t.Errorf("got %+v, want /review", d)
	}
	if d.Confidence < 0.9 {
		t.Errorf("expected high confidence for axis-aligned prompt, got %v", d.Confidence)
	}
	if d.Margin <= 0 {
		t.Errorf("expected positive margin, got %v", d.Margin)
	}
}

func TestSemantic_MultiVectorMaxWins(t *testing.T) {
	// "look at this diff" is closer to its exemplar than to the
	// skill's other exemplars; the max-cosine rule must still pick
	// /review.
	emb := newFakeEmb()
	emb.vectors["just look at this diff"] = unit(0.95, 0.1, 0, 0)
	s, _ := NewSemantic(context.Background(), emb, newTestCatalog())
	d, _ := s.Route(context.Background(), "just look at this diff")
	if d.Skill != "/review" {
		t.Errorf("multi-vector max-cosine failed: got %+v", d)
	}
}

func TestSemantic_BelowMinScoreFallsThrough(t *testing.T) {
	emb := newFakeEmb()
	s, _ := NewSemantic(context.Background(), emb, newTestCatalog(), WithMinScore(0.5))
	d, _ := s.Route(context.Background(), "unrelated random text")
	if !d.FellThrough {
		t.Errorf("expected fellthrough for unrelated prompt, got %+v", d)
	}
}

func TestSemantic_NarrowMarginReturned(t *testing.T) {
	emb := newFakeEmb()
	s, _ := NewSemantic(context.Background(), emb, newTestCatalog())
	d, _ := s.Route(context.Background(), "close-call: review or sec")
	if d.FellThrough {
		t.Fatalf("close-call should not fellthrough above MinScore, got %+v", d)
	}
	if d.Margin > 0.15 {
		t.Errorf("expected narrow margin on close call, got %v", d.Margin)
	}
}

func TestSemantic_RankTopK(t *testing.T) {
	emb := newFakeEmb()
	s, _ := NewSemantic(context.Background(), emb, newTestCatalog())
	r, err := s.Rank(context.Background(), "please review my PR", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 2 {
		t.Fatalf("want 2 entries, got %d", len(r))
	}
	if r[0].Skill != "/review" {
		t.Errorf("want /review first, got %s", r[0].Skill)
	}
	if r[0].Score < r[1].Score {
		t.Errorf("ranking not descending: %+v", r)
	}
}

func TestSemantic_RankAllWhenKZero(t *testing.T) {
	emb := newFakeEmb()
	s, _ := NewSemantic(context.Background(), emb, newTestCatalog())
	r, _ := s.Rank(context.Background(), "please review my PR", 0)
	if len(r) != 3 {
		t.Errorf("k=0 should return full catalog, got %d", len(r))
	}
}

func TestSemantic_ExemplarFallbackDescription(t *testing.T) {
	// Skill with only Description: must still embed something.
	emb := &fakeEmbedder{
		dim: 3,
		vectors: map[string][]float32{
			"Review a pull request": unit(1, 0, 0),
			"please review my PR":   unit(0.95, 0.05, 0),
		},
	}
	skills := []Skill{
		{Name: "/review", Description: "Review a pull request"},
		{Name: "/init", Description: "Initialize"},
	}
	s, err := NewSemantic(context.Background(), emb, skills)
	if err != nil {
		t.Fatal(err)
	}
	d, _ := s.Route(context.Background(), "please review my PR")
	if d.Skill != "/review" {
		t.Errorf("description-only exemplar failed: %+v", d)
	}
}

func TestSemantic_ExemplarFallbackName(t *testing.T) {
	emb := &fakeEmbedder{
		dim: 3,
		vectors: map[string][]float32{
			"/review":             unit(1, 0, 0),
			"please review my PR": unit(0.95, 0.05, 0),
		},
	}
	skills := []Skill{{Name: "/review"}}
	s, err := NewSemantic(context.Background(), emb, skills)
	if err != nil {
		t.Fatal(err)
	}
	d, _ := s.Route(context.Background(), "please review my PR")
	if d.Skill != "/review" {
		t.Errorf("name-only fallback failed: %+v", d)
	}
}

func TestSemantic_RejectsEmptyCatalog(t *testing.T) {
	emb := newFakeEmb()
	_, err := NewSemantic(context.Background(), emb, nil)
	if err == nil {
		t.Fatal("expected error on empty catalog")
	}
}

func TestSemantic_RejectsNilEmbedder(t *testing.T) {
	_, err := NewSemantic(context.Background(), nil, newTestCatalog())
	if err == nil {
		t.Fatal("expected error on nil embedder")
	}
}

func TestSemantic_RejectsSkillWithNoText(t *testing.T) {
	emb := newFakeEmb()
	skills := []Skill{{Name: "", Description: "", Examples: nil}}
	_, err := NewSemantic(context.Background(), emb, skills)
	if err == nil {
		t.Fatal("expected error on skill with no usable text")
	}
}

func TestSemantic_PropagatesEmbedError(t *testing.T) {
	emb := &fakeEmbedder{dim: 3, err: errors.New("ONNX boom")}
	_, err := NewSemantic(context.Background(), emb, newTestCatalog())
	if err == nil {
		t.Fatal("expected construction to propagate embed error")
	}
}
