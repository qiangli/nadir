package budget

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/qiangli/nadir/internal/modelmeta"
)

func TestAllowedAndRecord(t *testing.T) {
	tr := New(Config{DailyLimitUSD: 1.0}, modelmeta.Default())
	if ok, _ := tr.Allowed(); !ok {
		t.Fatal("initial state should allow")
	}
	tr.Record("gpt", 0.50)
	if ok, _ := tr.Allowed(); !ok {
		t.Fatal("half-budget should still allow")
	}
	tr.Record("gpt", 0.60)
	ok, reason := tr.Allowed()
	if ok || reason == "" {
		t.Fatalf("over-budget should deny: ok=%v reason=%q", ok, reason)
	}
}

func TestAlertHookFires(t *testing.T) {
	fired := 0
	tr := New(Config{DailyLimitUSD: 1.0}, nil)
	tr.AddHook(func(_ AlertEvent) { fired++ })
	tr.Record("m", 0.51) // crosses 50%
	tr.Record("m", 0.30) // crosses 80%
	tr.Record("m", 0.30) // crosses 100%
	if fired != 3 {
		t.Errorf("expected 3 alerts, got %d", fired)
	}
}

func TestDailyRollover(t *testing.T) {
	now := time.Date(2026, 1, 1, 23, 59, 0, 0, time.UTC)
	tr := New(Config{DailyLimitUSD: 1.0, NowFunc: func() time.Time { return now }}, nil)
	tr.Record("m", 0.90)

	now = now.Add(2 * time.Minute) // next day
	if got := tr.Snapshot().DailyUSD; got != 0 {
		t.Errorf("daily should roll, got %v", got)
	}
	if got := tr.Snapshot().MonthlyUSD; got != 0.90 {
		t.Errorf("monthly should NOT roll, got %v", got)
	}
}

func TestSaveLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "b.json")
	tr := New(Config{}, nil)
	tr.Record("m", 0.42)
	if err := tr.SaveTo(path); err != nil {
		t.Fatal(err)
	}
	tr2 := New(Config{}, nil)
	if err := tr2.LoadFrom(path); err != nil {
		t.Fatal(err)
	}
	if got := tr2.Snapshot().TotalUSD; got != 0.42 {
		t.Errorf("reloaded total = %v", got)
	}
}
