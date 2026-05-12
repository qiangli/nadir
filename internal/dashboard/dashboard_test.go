package dashboard

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/qiangli/nadir/internal/budget"
	"github.com/qiangli/nadir/internal/store/sqlite"
	"github.com/qiangli/nadir/modelmeta"
	"github.com/qiangli/nadir/types"
)

func TestDashboardRendersAllSections(t *testing.T) {
	dir := t.TempDir()
	store, err := sqlite.Open(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	_ = store.Log(context.Background(), &types.RequestEntry{
		ID: "1", Timestamp: time.Now(),
		Model: "haiku", Tier: types.TierSimple, Provider: "anthropic",
		PromptTokens: 100, CompletionTokens: 50, CostUSD: 0.05, LatencyMs: 200, Status: "ok",
	})
	tr := budget.New(budget.Config{DailyLimitUSD: 10}, modelmeta.Default())
	tr.Record("haiku", 0.05)

	h := Handler(Deps{Store: store, Budget: tr, Table: modelmeta.Default()})
	r := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d", w.Code)
	}
	body, _ := io.ReadAll(w.Body)
	out := string(body)
	for _, want := range []string{"<h1>nadir dashboard</h1>", "budget", "last 24h by model", "haiku"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in body", want)
		}
	}
}

func TestDashboardEmptyStoreNoError(t *testing.T) {
	dir := t.TempDir()
	store, _ := sqlite.Open(filepath.Join(dir, "test.db"))
	defer store.Close()
	tr := budget.New(budget.Config{}, nil)

	h := Handler(Deps{Store: store, Budget: tr, Table: modelmeta.Default()})
	r := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("empty dashboard = %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "no requests yet") {
		t.Error("empty state message missing")
	}
}

func TestDashboardEscapesModelNames(t *testing.T) {
	// HTML injection check: a model name containing < should be escaped.
	dir := t.TempDir()
	store, _ := sqlite.Open(filepath.Join(dir, "test.db"))
	defer store.Close()
	_ = store.Log(context.Background(), &types.RequestEntry{
		ID: "1", Timestamp: time.Now(),
		Model: "<script>", Tier: types.TierSimple, Provider: "x", Status: "ok",
	})
	tr := budget.New(budget.Config{}, nil)

	h := Handler(Deps{Store: store, Budget: tr, Table: modelmeta.Default()})
	r := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if strings.Contains(w.Body.String(), "<script>x</script>") {
		t.Error("model name should be HTML-escaped")
	}
}
