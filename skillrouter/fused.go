package skillrouter

import (
	"context"
	"errors"
	"sort"
)

// FusedSemantic combines multiple Semantic matchers via Reciprocal
// Rank Fusion (RRF) — a classic IR technique for merging rankings
// from heterogeneous scorers. Unlike Ensemble (which fuses at the
// vector level by concatenation and produces a cosine *average*), RRF
// operates on per-matcher *rankings*: a skill that scores #1 in one
// matcher and #3 in another beats a skill that's #2 in both.
//
// This is the right fusion when the underlying matchers use different
// score scales or have different sensitivities (TF-IDF returns sparse
// high-magnitude scores when vocabulary overlaps; HashEmbedder returns
// dense moderate scores from subword similarity). RRF normalizes both
// to a common rank-based scale.
//
// Formula (Cormack et al., 2009): for each skill s, RRF score =
// sum over matchers of 1 / (k + rank(s)), where rank is 0-indexed and
// k=60 is the standard damping constant. The skill with the highest
// fused score wins.
//
// Confidence is the top-1 fused score (theoretical max for n matchers
// is n/k). Margin is top1-top2 in the fused space.
//
// FellThrough: triggered when every matcher reports a top-1 below its
// own MinScore — i.e., none of the underlying signals consider any
// skill viable. Less aggressive than per-matcher fallthrough, which
// is the right move: if either method finds a credible match, we
// should consider it.
type FusedSemantic struct {
	matchers []*Semantic
	k        float64
}

// FusedOption configures a FusedSemantic.
type FusedOption func(*FusedSemantic)

// WithRRFK sets the RRF damping constant. Default 60, the value
// recommended by Cormack et al. Higher k flattens the rank-vs-score
// curve (later ranks contribute more); lower k makes top ranks
// dominate. The default works well across a wide range of catalog
// sizes.
func WithRRFK(k float64) FusedOption {
	return func(f *FusedSemantic) {
		if k > 0 {
			f.k = k
		}
	}
}

// NewFusedSemantic wires two or more Semantic matchers into one. All
// matchers must have been constructed over the same skill catalog —
// the fusion keys on skill Name. Returns an error on fewer than 2
// matchers (would degenerate to passthrough).
func NewFusedSemantic(matchers []*Semantic, opts ...FusedOption) (*FusedSemantic, error) {
	if len(matchers) < 2 {
		return nil, errors.New("skillrouter: FusedSemantic needs at least 2 matchers")
	}
	for i, m := range matchers {
		if m == nil {
			return nil, errors.New("skillrouter: nil matcher in FusedSemantic")
		}
		// Sanity-check catalog alignment: all matchers should expose
		// the same skill names. We only check counts here; mismatched
		// names will produce 0-weighted entries downstream.
		if i > 0 && len(m.skills) != len(matchers[0].skills) {
			return nil, errors.New("skillrouter: FusedSemantic matchers have mismatched catalog sizes")
		}
	}
	return &FusedSemantic{matchers: matchers, k: 60}, nil
}

// Route runs every underlying matcher, fuses by RRF, and returns the
// winning skill. Each underlying Rank call embeds the prompt once;
// total work is len(matchers) × per-Embed cost.
func (f *FusedSemantic) Route(ctx context.Context, prompt string) (*Decision, error) {
	allRanked := make([][]Ranked, len(f.matchers))
	allUnviable := true
	for i, m := range f.matchers {
		r, err := m.Rank(ctx, prompt, 0)
		if err != nil {
			return nil, err
		}
		allRanked[i] = r
		// Track whether ANY matcher saw a viable top-1; otherwise
		// fallthrough early.
		if len(r) > 0 && r[0].Score >= m.minScore {
			allUnviable = false
		}
	}
	if allUnviable {
		return &Decision{FellThrough: true}, nil
	}

	// RRF: per-skill score = sum over matchers of 1/(k + rank).
	fused := make(map[string]float64)
	for _, ranked := range allRanked {
		for rank, r := range ranked {
			fused[r.Skill] += 1.0 / (f.k + float64(rank))
		}
	}

	// Sort skills by fused score, descending.
	type entry struct {
		Skill string
		Score float64
	}
	entries := make([]entry, 0, len(fused))
	for s, sc := range fused {
		entries = append(entries, entry{Skill: s, Score: sc})
	}
	sort.SliceStable(entries, func(i, j int) bool { return entries[i].Score > entries[j].Score })

	if len(entries) == 0 {
		return &Decision{FellThrough: true}, nil
	}
	top := entries[0]
	margin := 0.0
	if len(entries) > 1 {
		margin = top.Score - entries[1].Score
	}
	return &Decision{
		Skill:      top.Skill,
		Confidence: top.Score,
		Margin:     margin,
	}, nil
}

var _ Matcher = (*FusedSemantic)(nil)
