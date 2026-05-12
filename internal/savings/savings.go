// Package savings computes how much money the router saved compared
// to a hypothetical "always premium" baseline: replay every logged
// request through a baseline model's cost, subtract actual spend, and
// report the delta + savings ratio.
package savings

import (
	"context"
	"time"

	"github.com/qiangli/nadir/internal/modelmeta"
	"github.com/qiangli/nadir/internal/store/sqlite"
)

type Report struct {
	BaselineModel  string  `json:"baseline_model"`
	BaselineUSD    float64 `json:"baseline_usd"`
	ActualUSD      float64 `json:"actual_usd"`
	SavedUSD       float64 `json:"saved_usd"`
	SavedFraction  float64 `json:"saved_fraction"`
	RequestCount   int     `json:"request_count"`
	Since          time.Time `json:"since"`
}

type Calculator struct {
	store *sqlite.Store
	table *modelmeta.Table
}

func New(store *sqlite.Store, table *modelmeta.Table) *Calculator {
	return &Calculator{store: store, table: table}
}

// Compute walks logged rows since `since` and totals actual vs
// baseline cost. The baseline is the most-expensive model in the
// modelmeta table by default, or whatever the caller passes.
func (c *Calculator) Compute(ctx context.Context, since time.Time, baseline string) (*Report, error) {
	if baseline == "" {
		baseline = c.mostExpensive()
	}
	rows, err := c.store.Query(ctx, sqlite.QueryFilter{Since: since, Limit: 100000})
	if err != nil {
		return nil, err
	}
	out := &Report{BaselineModel: baseline, Since: since, RequestCount: len(rows)}
	for _, r := range rows {
		out.ActualUSD += r.CostUSD
		out.BaselineUSD += c.table.Cost(baseline, r.PromptTokens, r.CompletionTokens)
	}
	out.SavedUSD = out.BaselineUSD - out.ActualUSD
	if out.BaselineUSD > 0 {
		out.SavedFraction = out.SavedUSD / out.BaselineUSD
	}
	return out, nil
}

func (c *Calculator) mostExpensive() string {
	all := c.table.All()
	if len(all) == 0 {
		return ""
	}
	// All() sorts cheapest first; last entry is the most expensive.
	return all[len(all)-1].Name
}
