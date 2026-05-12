package metrics

import (
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestObserveAndScrape(t *testing.T) {
	c := New()
	c.Observe("haiku", "simple", "ok", 123*time.Millisecond, 10, 20, 0.001)
	c.CacheHits.Inc()
	c.Fallbacks.WithLabelValues("haiku", "sonnet").Inc()

	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/metrics", nil)
	c.Handler().ServeHTTP(rr, req)

	body, _ := io.ReadAll(rr.Body)
	out := string(body)
	for _, want := range []string{
		"nadir_requests_total",
		"nadir_request_latency_ms",
		"nadir_prompt_tokens_total",
		"nadir_completion_tokens_total",
		"nadir_cost_usd_total",
		"nadir_cache_hits_total",
		"nadir_fallback_total",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("metrics scrape missing %q", want)
		}
	}
}
