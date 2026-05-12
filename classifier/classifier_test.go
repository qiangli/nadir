package classifier

import (
	"context"
	"testing"

	"github.com/qiangli/nadir/types"
)

func TestHeuristicClassifyShortPrompt(t *testing.T) {
	c := NewHeuristic(DefaultThresholds())
	tier, score, _, err := c.Classify(context.Background(), "hi")
	if err != nil {
		t.Fatalf("classify: %v", err)
	}
	if tier != types.TierSimple {
		t.Errorf("short prompt → got tier %s, want simple", tier)
	}
	if score > 0.35 {
		t.Errorf("short prompt → score %v > 0.35", score)
	}
}

func TestHeuristicClassifyComplexPrompt(t *testing.T) {
	c := NewHeuristic(DefaultThresholds())
	prompt := "Refactor this concurrent code to eliminate the race condition. " +
		"```go\nvar mu sync.Mutex\nfunc do() { mu.Lock(); defer mu.Unlock() }\n```\n" +
		"Walk me through your reasoning step by step, and explain why the original deadlocks."
	tier, score, _, err := c.Classify(context.Background(), prompt)
	if err != nil {
		t.Fatalf("classify: %v", err)
	}
	if tier != types.TierComplex {
		t.Errorf("complex prompt → got tier %s (score %v), want complex", tier, score)
	}
}

func TestBucketScoreBoundaries(t *testing.T) {
	t1 := Thresholds{Simple: 0.35, Complex: 0.65}
	tests := []struct {
		score float64
		want  types.Tier
	}{
		{0.0, types.TierSimple},
		{0.35, types.TierSimple},
		{0.36, types.TierSimple}, // no mid by default
		{0.64, types.TierSimple},
		{0.65, types.TierComplex},
		{1.0, types.TierComplex},
	}
	for _, tc := range tests {
		if got := BucketScore(tc.score, t1); got != tc.want {
			t.Errorf("BucketScore(%v) = %s, want %s", tc.score, got, tc.want)
		}
	}
}

func TestBucketScoreWithMidTier(t *testing.T) {
	t1 := Thresholds{Simple: 0.35, Complex: 0.65, HasMid: true}
	if got := BucketScore(0.5, t1); got != types.TierMid {
		t.Errorf("mid tier 0.5 = %s, want mid", got)
	}
}
