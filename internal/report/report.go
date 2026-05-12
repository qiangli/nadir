// Package report renders text and HTML usage reports from the SQLite
// request log. It is the read side of the request_logger.py output:
// per-model and per-day breakdowns with cost, token, and latency
// summaries.
package report

import (
	"context"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/qiangli/nadir/internal/store/sqlite"
)

type Renderer struct {
	store *sqlite.Store
}

func New(store *sqlite.Store) *Renderer { return &Renderer{store: store} }

type Options struct {
	Since   time.Duration // lookback window; 0 → 24h
	ByModel bool
	ByDay   bool
	HTML    bool
}

func (r *Renderer) Render(ctx context.Context, w io.Writer, opts Options) error {
	if opts.Since == 0 {
		opts.Since = 24 * time.Hour
	}
	since := time.Now().Add(-opts.Since)

	byModel, err := r.store.AggregateByModel(ctx, since)
	if err != nil {
		return err
	}
	byDay, err := r.store.AggregateByDay(ctx, since)
	if err != nil {
		return err
	}

	if opts.HTML {
		return renderHTML(w, since, byModel, byDay, opts)
	}
	return renderText(w, since, byModel, byDay, opts)
}

func renderText(w io.Writer, since time.Time, byModel, byDay []sqlite.Aggregate, opts Options) error {
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintf(tw, "nadir report — since %s\n\n", since.Format(time.RFC3339))

	if opts.ByModel || (!opts.ByModel && !opts.ByDay) {
		fmt.Fprintln(tw, "BY MODEL")
		fmt.Fprintln(tw, "model\tcount\tprompt_tok\tcompl_tok\tcost_usd\tavg_lat_ms")
		var totalCost float64
		var totalCount int
		for _, a := range byModel {
			fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%.4f\t%.0f\n",
				a.Model, a.Count, a.PromptTokens, a.CompletionTokens, a.CostUSD, a.AvgLatencyMs)
			totalCost += a.CostUSD
			totalCount += a.Count
		}
		fmt.Fprintf(tw, "TOTAL\t%d\t\t\t%.4f\t\n\n", totalCount, totalCost)
	}

	if opts.ByDay {
		fmt.Fprintln(tw, "BY DAY")
		fmt.Fprintln(tw, "day\tcount\tprompt_tok\tcompl_tok\tcost_usd\tavg_lat_ms")
		for _, a := range byDay {
			fmt.Fprintf(tw, "%s\t%d\t%d\t%d\t%.4f\t%.0f\n",
				a.Day, a.Count, a.PromptTokens, a.CompletionTokens, a.CostUSD, a.AvgLatencyMs)
		}
		fmt.Fprintln(tw)
	}
	return tw.Flush()
}

func renderHTML(w io.Writer, since time.Time, byModel, byDay []sqlite.Aggregate, _ Options) error {
	var b strings.Builder
	b.WriteString(`<!DOCTYPE html><html><head><title>nadir report</title>`)
	b.WriteString(`<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;color:#222}table{border-collapse:collapse;width:100%}th,td{padding:.5em;border-bottom:1px solid #ddd;text-align:left}th{background:#f7f7f7}</style>`)
	b.WriteString(`</head><body>`)
	fmt.Fprintf(&b, "<h1>nadir report</h1><p>since %s</p>", since.Format(time.RFC3339))

	b.WriteString("<h2>by model</h2><table><thead><tr><th>model</th><th>count</th><th>prompt_tok</th><th>compl_tok</th><th>cost_usd</th><th>avg_lat_ms</th></tr></thead><tbody>")
	for _, a := range byModel {
		fmt.Fprintf(&b, "<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%.4f</td><td>%.0f</td></tr>",
			a.Model, a.Count, a.PromptTokens, a.CompletionTokens, a.CostUSD, a.AvgLatencyMs)
	}
	b.WriteString("</tbody></table>")

	if len(byDay) > 0 {
		b.WriteString("<h2>by day</h2><table><thead><tr><th>day</th><th>count</th><th>prompt_tok</th><th>compl_tok</th><th>cost_usd</th><th>avg_lat_ms</th></tr></thead><tbody>")
		for _, a := range byDay {
			fmt.Fprintf(&b, "<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%.4f</td><td>%.0f</td></tr>",
				a.Day, a.Count, a.PromptTokens, a.CompletionTokens, a.CostUSD, a.AvgLatencyMs)
		}
		b.WriteString("</tbody></table>")
	}
	b.WriteString("</body></html>")
	_, err := io.WriteString(w, b.String())
	return err
}
