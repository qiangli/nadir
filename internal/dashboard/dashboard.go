// Package dashboard serves a simple HTML overview at /dashboard:
// recent traffic by model, current budget snapshot, and a savings
// summary. It reads from the SQLite store and the budget tracker
// directly — no live state needed.
//
// The HTML is intentionally tiny (single-file, embedded CSS, no JS)
// so a fresh install renders instantly with no external assets.
package dashboard

import (
	"context"
	"fmt"
	"html"
	"net/http"
	"time"

	"github.com/qiangli/nadir/internal/budget"
	"github.com/qiangli/nadir/modelmeta"
	"github.com/qiangli/nadir/internal/savings"
	"github.com/qiangli/nadir/internal/store/sqlite"
)

type Deps struct {
	Store  *sqlite.Store
	Budget *budget.Tracker
	Table  *modelmeta.Table
}

func Handler(d Deps) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
		defer cancel()

		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, prelude)
		fmt.Fprint(w, "<h1>nadir dashboard</h1>")
		renderBudget(w, d.Budget)
		renderTraffic(ctx, w, d.Store)
		renderSavings(ctx, w, d.Store, d.Table)
		fmt.Fprint(w, postlude)
	})
}

const prelude = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>nadir</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;color:#222;padding:0 1rem}
h1{font-size:1.6rem}h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{padding:.4rem .6rem;border-bottom:1px solid #eee;text-align:left}
th{background:#fafafa}
.kv{display:flex;gap:1.5rem;flex-wrap:wrap;margin:1rem 0}
.kv div{background:#f5f5f5;padding:.6rem 1rem;border-radius:6px}
.kv label{display:block;font-size:.75rem;text-transform:uppercase;color:#666;letter-spacing:.05em}
.kv strong{font-size:1.1rem}
</style></head><body>`

const postlude = `</body></html>`

func renderBudget(w http.ResponseWriter, b *budget.Tracker) {
	if b == nil {
		return
	}
	s := b.Snapshot()
	fmt.Fprintf(w, `<h2>budget</h2><div class="kv">
		<div><label>today</label><strong>$%.4f</strong></div>
		<div><label>this month</label><strong>$%.4f</strong></div>
		<div><label>total</label><strong>$%.4f</strong></div>
	</div>`, s.DailyUSD, s.MonthlyUSD, s.TotalUSD)
}

func renderTraffic(ctx context.Context, w http.ResponseWriter, st *sqlite.Store) {
	if st == nil {
		return
	}
	since := time.Now().Add(-24 * time.Hour)
	agg, err := st.AggregateByModel(ctx, since)
	if err != nil {
		fmt.Fprintf(w, "<p>(traffic unavailable: %s)</p>", html.EscapeString(err.Error()))
		return
	}
	fmt.Fprint(w, "<h2>last 24h by model</h2>")
	if len(agg) == 0 {
		fmt.Fprint(w, "<p>(no requests yet)</p>")
		return
	}
	fmt.Fprint(w, "<table><thead><tr><th>model</th><th>count</th><th>prompt_tok</th><th>compl_tok</th><th>cost_usd</th><th>avg_lat_ms</th></tr></thead><tbody>")
	for _, a := range agg {
		fmt.Fprintf(w, "<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%.4f</td><td>%.0f</td></tr>",
			html.EscapeString(a.Model), a.Count, a.PromptTokens, a.CompletionTokens, a.CostUSD, a.AvgLatencyMs)
	}
	fmt.Fprint(w, "</tbody></table>")
}

func renderSavings(ctx context.Context, w http.ResponseWriter, st *sqlite.Store, table *modelmeta.Table) {
	if st == nil || table == nil {
		return
	}
	calc := savings.New(st, table)
	rep, err := calc.Compute(ctx, time.Now().Add(-7*24*time.Hour), "")
	if err != nil || rep.RequestCount == 0 {
		return
	}
	fmt.Fprintf(w, `<h2>savings (last 7 days)</h2><div class="kv">
		<div><label>actual</label><strong>$%.4f</strong></div>
		<div><label>baseline (%s)</label><strong>$%.4f</strong></div>
		<div><label>saved</label><strong>$%.4f (%.0f%%)</strong></div>
	</div>`, rep.ActualUSD, html.EscapeString(rep.BaselineModel), rep.BaselineUSD, rep.SavedUSD, rep.SavedFraction*100)
}
