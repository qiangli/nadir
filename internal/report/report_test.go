package report

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/qiangli/nadir/internal/store/sqlite"
	"github.com/qiangli/nadir/internal/types"
)

func newSeededStore(t *testing.T) *sqlite.Store {
	t.Helper()
	s, err := sqlite.Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	now := time.Now()
	for i, m := range []string{"haiku", "haiku", "sonnet", "opus"} {
		_ = s.Log(context.Background(), &types.RequestEntry{
			ID: "id" + string(rune('a'+i)), Timestamp: now,
			Model: m, Tier: types.TierSimple, Provider: "openai",
			PromptTokens: 10 * (i + 1), CompletionTokens: 5 * (i + 1),
			CostUSD: 0.001 * float64(i+1), LatencyMs: int64(100 * (i + 1)),
			Status: "ok",
		})
	}
	return s
}

func TestRenderTextHasByModelTable(t *testing.T) {
	s := newSeededStore(t)
	r := New(s)
	var buf bytes.Buffer
	if err := r.Render(context.Background(), &buf, Options{Since: time.Hour, ByModel: true}); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	for _, want := range []string{"BY MODEL", "haiku", "sonnet", "opus", "TOTAL"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\n%s", want, out)
		}
	}
}

func TestRenderHTMLHasTableTags(t *testing.T) {
	s := newSeededStore(t)
	r := New(s)
	var buf bytes.Buffer
	if err := r.Render(context.Background(), &buf, Options{Since: time.Hour, ByModel: true, HTML: true}); err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, "<table") {
		t.Errorf("HTML output missing <table: %s", out[:200])
	}
	if !strings.Contains(out, "haiku") {
		t.Errorf("HTML output missing model name")
	}
}

func TestRenderByDay(t *testing.T) {
	s := newSeededStore(t)
	r := New(s)
	var buf bytes.Buffer
	if err := r.Render(context.Background(), &buf, Options{Since: time.Hour, ByDay: true}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(buf.String(), "BY DAY") {
		t.Errorf("missing BY DAY section: %s", buf.String())
	}
}

func TestRenderEmptyStore(t *testing.T) {
	// Empty DB: no rows but report should still render without erroring.
	s, _ := sqlite.Open(filepath.Join(t.TempDir(), "empty.db"))
	defer s.Close()
	var buf bytes.Buffer
	if err := New(s).Render(context.Background(), &buf, Options{Since: time.Hour, ByModel: true}); err != nil {
		t.Errorf("empty render failed: %v", err)
	}
}
