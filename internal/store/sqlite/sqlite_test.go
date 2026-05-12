package sqlite

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/qiangli/nadir/types"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.db")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestStoreLogAndQuery(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	now := time.Now()
	for i := range 3 {
		err := s.Log(ctx, &types.RequestEntry{
			ID:               "id-" + time.Duration(i).String(),
			Timestamp:        now.Add(time.Duration(i) * time.Second),
			Model:            "gpt-test",
			Tier:             types.TierSimple,
			Provider:         "openai",
			PromptTokens:     10,
			CompletionTokens: 20,
			CostUSD:          0.001,
			LatencyMs:        100,
			Status:           "ok",
			UserID:           "u1",
			Score:            0.2,
			Modifiers:        []string{"classified"},
		})
		if err != nil {
			t.Fatalf("log %d: %v", i, err)
		}
	}
	n, err := s.Count(ctx)
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if n != 3 {
		t.Errorf("count = %d, want 3", n)
	}
	rows, err := s.Query(ctx, QueryFilter{})
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if len(rows) != 3 {
		t.Errorf("rows = %d", len(rows))
	}
}

func TestStoreAggregates(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	now := time.Now()
	for i, model := range []string{"a", "a", "b"} {
		_ = s.Log(ctx, &types.RequestEntry{
			ID: "row-" + model + time.Duration(i).String(), Timestamp: now,
			Model: model, Tier: types.TierSimple, Provider: "openai",
			CostUSD: 0.01, LatencyMs: 100, Status: "ok",
		})
	}
	agg, err := s.AggregateByModel(ctx, now.Add(-time.Hour))
	if err != nil {
		t.Fatalf("agg: %v", err)
	}
	if len(agg) != 2 {
		t.Fatalf("want 2 model groups, got %d", len(agg))
	}
	// Highest-spend model first.
	if agg[0].Model != "a" {
		t.Errorf("expected 'a' first (count 2), got %s", agg[0].Model)
	}
}
