// Package metrics exposes Prometheus counters and histograms covering
// the proxy hot path: request count by (model, tier, status), latency
// histogram, token + cost totals, cache hits, fallback count.
//
// The collectors are registered to a private Registry so /metrics
// only emits nadir's series — runtime Go stats are intentionally
// excluded to keep the scrape payload tight.
package metrics

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type Collectors struct {
	reg *prometheus.Registry

	Requests       *prometheus.CounterVec
	LatencyMs      *prometheus.HistogramVec
	PromptTokens   *prometheus.CounterVec
	CompletionTokens *prometheus.CounterVec
	CostUSD        *prometheus.CounterVec
	CacheHits      prometheus.Counter
	CacheMisses    prometheus.Counter
	Fallbacks      *prometheus.CounterVec
}

func New() *Collectors {
	reg := prometheus.NewRegistry()
	c := &Collectors{
		reg: reg,
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "nadir_requests_total",
			Help: "Total chat-completion requests handled.",
		}, []string{"model", "tier", "status"}),
		LatencyMs: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "nadir_request_latency_ms",
			Help:    "End-to-end request latency in milliseconds.",
			Buckets: []float64{10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000},
		}, []string{"model", "tier"}),
		PromptTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "nadir_prompt_tokens_total",
			Help: "Total prompt tokens sent upstream.",
		}, []string{"model"}),
		CompletionTokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "nadir_completion_tokens_total",
			Help: "Total completion tokens received.",
		}, []string{"model"}),
		CostUSD: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "nadir_cost_usd_total",
			Help: "Total estimated USD spend.",
		}, []string{"model"}),
		CacheHits: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "nadir_cache_hits_total",
			Help: "Prompt cache hits.",
		}),
		CacheMisses: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "nadir_cache_misses_total",
			Help: "Prompt cache misses.",
		}),
		Fallbacks: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "nadir_fallback_total",
			Help: "Times a fallback model was used.",
		}, []string{"primary", "fallback"}),
	}
	reg.MustRegister(c.Requests, c.LatencyMs, c.PromptTokens, c.CompletionTokens, c.CostUSD, c.CacheHits, c.CacheMisses, c.Fallbacks)
	return c
}

// Handler returns the /metrics http.Handler bound to nadir's private
// registry (no Go-runtime stats).
func (c *Collectors) Handler() http.Handler {
	return promhttp.HandlerFor(c.reg, promhttp.HandlerOpts{})
}

// Observe is a one-shot helper the proxy worker pool calls after a
// request completes.
func (c *Collectors) Observe(model, tier, status string, latency time.Duration, promptTokens, completionTokens int, cost float64) {
	c.Requests.WithLabelValues(model, tier, status).Inc()
	c.LatencyMs.WithLabelValues(model, tier).Observe(float64(latency.Milliseconds()))
	if promptTokens > 0 {
		c.PromptTokens.WithLabelValues(model).Add(float64(promptTokens))
	}
	if completionTokens > 0 {
		c.CompletionTokens.WithLabelValues(model).Add(float64(completionTokens))
	}
	if cost > 0 {
		c.CostUSD.WithLabelValues(model).Add(cost)
	}
}
