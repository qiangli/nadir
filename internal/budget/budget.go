// Package budget tracks per-user and global spend, gates requests
// against daily/monthly limits, and fires alerts on threshold cross.
// State is held in memory and optionally flushed to disk on demand
// (callers wire ~/.nadir/logs/budget_state.json via SaveTo/LoadFrom).
package budget

import (
	"encoding/json"
	"errors"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/qiangli/nadir/modelmeta"
	"github.com/qiangli/nadir/types"
)

func formatF(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}

type Config struct {
	DailyLimitUSD   float64   // 0 = no limit
	MonthlyLimitUSD float64   // 0 = no limit
	AlertThresholds []float64 // fractions of limit (e.g. 0.5, 0.8, 1.0)
	NowFunc         func() time.Time
}

type Tracker struct {
	cfg   Config
	table *modelmeta.Table

	mu    sync.Mutex
	state State
	fired map[string]bool // dedupe alerts per (period+threshold)
	hooks []AlertHook
}

type State struct {
	Day        string  `json:"day"`   // YYYY-MM-DD
	Month      string  `json:"month"` // YYYY-MM
	DailyUSD   float64 `json:"daily_usd"`
	MonthlyUSD float64 `json:"monthly_usd"`
	TotalUSD   float64 `json:"total_usd"`
}

type AlertHook func(event AlertEvent)

type AlertEvent struct {
	Period    string // "daily" | "monthly"
	Threshold float64
	Limit     float64
	Spent     float64
	At        time.Time
}

func New(cfg Config, table *modelmeta.Table) *Tracker {
	if cfg.NowFunc == nil {
		cfg.NowFunc = time.Now
	}
	if len(cfg.AlertThresholds) == 0 {
		cfg.AlertThresholds = []float64{0.5, 0.8, 1.0}
	}
	return &Tracker{cfg: cfg, table: table, fired: make(map[string]bool)}
}

func (t *Tracker) AddHook(h AlertHook) {
	t.mu.Lock()
	t.hooks = append(t.hooks, h)
	t.mu.Unlock()
}

func (t *Tracker) Allowed() (bool, string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rollIfStaleLocked()
	if t.cfg.DailyLimitUSD > 0 && t.state.DailyUSD >= t.cfg.DailyLimitUSD {
		return false, "daily budget exceeded"
	}
	if t.cfg.MonthlyLimitUSD > 0 && t.state.MonthlyUSD >= t.cfg.MonthlyLimitUSD {
		return false, "monthly budget exceeded"
	}
	return true, ""
}

func (t *Tracker) Record(model string, cost float64) {
	if cost <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rollIfStaleLocked()
	t.state.DailyUSD += cost
	t.state.MonthlyUSD += cost
	t.state.TotalUSD += cost
	t.maybeAlertLocked()
	_ = model
}

func (t *Tracker) Estimate(model string, promptTokens, completionTokens int) float64 {
	if t.table == nil {
		return 0
	}
	return t.table.Cost(model, promptTokens, completionTokens)
}

func (t *Tracker) Snapshot() State {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rollIfStaleLocked()
	s := t.state
	return s
}

func (t *Tracker) rollIfStaleLocked() {
	now := t.cfg.NowFunc()
	day := now.Format("2006-01-02")
	month := now.Format("2006-01")
	if t.state.Day != day {
		t.state.DailyUSD = 0
		t.state.Day = day
		for k := range t.fired {
			if strings.HasPrefix(k, "daily-") {
				delete(t.fired, k)
			}
		}
	}
	if t.state.Month != month {
		t.state.MonthlyUSD = 0
		t.state.Month = month
		for k := range t.fired {
			if strings.HasPrefix(k, "monthly-") {
				delete(t.fired, k)
			}
		}
	}
}

func (t *Tracker) maybeAlertLocked() {
	for _, thr := range t.cfg.AlertThresholds {
		if t.cfg.DailyLimitUSD > 0 {
			key := "daily-" + formatPercent(thr)
			if !t.fired[key] && t.state.DailyUSD >= t.cfg.DailyLimitUSD*thr {
				t.fired[key] = true
				t.fireLocked(AlertEvent{Period: "daily", Threshold: thr, Limit: t.cfg.DailyLimitUSD, Spent: t.state.DailyUSD, At: t.cfg.NowFunc()})
			}
		}
		if t.cfg.MonthlyLimitUSD > 0 {
			key := "monthly-" + formatPercent(thr)
			if !t.fired[key] && t.state.MonthlyUSD >= t.cfg.MonthlyLimitUSD*thr {
				t.fired[key] = true
				t.fireLocked(AlertEvent{Period: "monthly", Threshold: thr, Limit: t.cfg.MonthlyLimitUSD, Spent: t.state.MonthlyUSD, At: t.cfg.NowFunc()})
			}
		}
	}
}

func (t *Tracker) fireLocked(ev AlertEvent) {
	for _, h := range t.hooks {
		h(ev)
	}
}

func formatPercent(f float64) string {
	return fmtFloat(f * 100)
}

func fmtFloat(f float64) string {
	// strconv.FormatFloat with -1 precision avoids trailing zeros
	// while preserving uniqueness across thresholds.
	return formatF(f)
}

// SaveTo writes the snapshot as JSON.
func (t *Tracker) SaveTo(path string) error {
	snap := t.Snapshot()
	b, err := json.MarshalIndent(snap, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// LoadFrom rehydrates a snapshot if the file exists; missing files are
// ignored so first-run is harmless.
func (t *Tracker) LoadFrom(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	var s State
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	t.mu.Lock()
	t.state = s
	t.mu.Unlock()
	return nil
}

var _ types.Budget = (*Tracker)(nil)
