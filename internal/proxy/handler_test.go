package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/qiangli/nadir/cache"
	"github.com/qiangli/nadir/classifier"
	"github.com/qiangli/nadir/internal/config"
	"github.com/qiangli/nadir/health"
	"github.com/qiangli/nadir/provider/fake"
	"github.com/qiangli/nadir/ratelimit"
	"github.com/qiangli/nadir/router"
	"github.com/qiangli/nadir/types"
)

type fixture struct {
	srv         *Server
	primary     *fake.OK
	fallback    *fake.OK
	errPrimary  *fake.Err
}

// newFixture builds a proxy with two providers ("primary", "fallback")
// and a config where primary=haiku, complex=opus, fallback chain
// configured to put fallback's model in the chain. Tests can replace
// the entries in srv.deps.Providers to drive different scenarios.
func newFixture(t *testing.T) *fixture {
	t.Helper()
	cfg := &config.Config{
		SimpleModel:    "haiku",
		MidModel:       "sonnet",
		ComplexModel:   "opus",
		FallbackChain:  []string{"sonnet"},
		TierThresholds: [2]float64{0.35, 0.65},
		UserRateLimit:  1000,
		UserRateWindow: 60_000_000_000,
		ProviderForModel: map[string]string{
			"haiku":  "primary",
			"sonnet": "fallback",
			"opus":   "primary",
		},
	}
	cls := classifier.NewHeuristic(classifier.Thresholds{Simple: 0.35, Complex: 0.65, HasMid: true})
	tracker := health.New()
	rt := router.New(cfg.RouterConfig(), cls, cache.NewSession(0), tracker)
	primary := &fake.OK{NameStr: "primary", Content: "hello from primary"}
	fallback := &fake.OK{NameStr: "fallback", Content: "hello from fallback"}

	srv := New(Deps{
		Router:       rt,
		Classifier:   cls,
		Providers:    map[string]types.LLMClient{"primary": primary, "fallback": fallback},
		PromptCache:  cache.NewPrompt(10, 0),
		UserLimiter:  ratelimit.NewUser(60_000_000_000, 1000),
		ModelLimiter: ratelimit.NewModel(),
		Health:       tracker,
		ModelToProvider: func(model string) string {
			if p, ok := cfg.ProviderForModel[model]; ok {
				return p
			}
			return "primary"
		},
	})
	return &fixture{srv: srv, primary: primary, fallback: fallback}
}

func doRequest(srv *Server, body any) *httptest.ResponseRecorder {
	b, _ := json.Marshal(body)
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(b))
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, r)
	return w
}

func TestHandlerRoutesSimplePromptToPrimary(t *testing.T) {
	f := newFixture(t)
	w := doRequest(f.srv, map[string]any{
		"model":    "auto",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("X-Routed-Model"); got != "haiku" {
		t.Errorf("X-Routed-Model=%q, want haiku", got)
	}
	if got := w.Header().Get("X-Routed-Tier"); got != "simple" {
		t.Errorf("X-Routed-Tier=%q, want simple", got)
	}
	if f.primary.Calls.Load() != 1 || f.fallback.Calls.Load() != 0 {
		t.Errorf("calls primary=%d fallback=%d, want 1/0", f.primary.Calls.Load(), f.fallback.Calls.Load())
	}
}

func TestHandlerFallbackOn429(t *testing.T) {
	f := newFixture(t)
	// Replace primary with a 429 fake.
	rateLimited := &fake.Err{NameStr: "primary", Kind: types.ErrRateLimit}
	f.srv.deps.Providers["primary"] = rateLimited
	f.errPrimary = rateLimited

	w := doRequest(f.srv, map[string]any{
		"model":    "auto",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 after fallback, got %d: %s", w.Code, w.Body.String())
	}
	if rateLimited.Calls.Load() == 0 {
		t.Error("primary should have been tried")
	}
	if f.fallback.Calls.Load() != 1 {
		t.Errorf("fallback should have been invoked once, got %d", f.fallback.Calls.Load())
	}
}

func TestHandlerFatalAuthAborts(t *testing.T) {
	f := newFixture(t)
	authErr := &fake.Err{NameStr: "primary", Kind: types.ErrAuth}
	f.srv.deps.Providers["primary"] = authErr

	w := doRequest(f.srv, map[string]any{
		"model":    "auto",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if w.Code != http.StatusUnauthorized {
		t.Errorf("auth error should surface as 401, got %d: %s", w.Code, w.Body.String())
	}
	if f.fallback.Calls.Load() != 0 {
		t.Error("fatal error must abort the chain — fallback should not be called")
	}
}

func TestHandlerBearerTokenEnforced(t *testing.T) {
	f := newFixture(t)
	f.srv.deps.AuthToken = "secret"

	// Without token.
	body, _ := json.Marshal(map[string]any{
		"model": "auto", "messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()
	f.srv.Handler().ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("no token = %d, want 401", w.Code)
	}

	// With token.
	r = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	r.Header.Set("Authorization", "Bearer secret")
	w = httptest.NewRecorder()
	f.srv.Handler().ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Errorf("with token = %d: %s", w.Code, w.Body.String())
	}
}

func TestHandlerStream(t *testing.T) {
	f := newFixture(t)
	w := doRequest(f.srv, map[string]any{
		"model":    "auto",
		"stream":   true,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type=%q", ct)
	}
	body, _ := io.ReadAll(w.Body)
	if !strings.Contains(string(body), "data:") {
		t.Errorf("body missing SSE frames: %q", body)
	}
	if !strings.Contains(string(body), "[DONE]") {
		t.Errorf("body missing [DONE] sentinel: %q", body)
	}
}

func TestHandlerStreamFallbackBeforeFirstChunk(t *testing.T) {
	f := newFixture(t)
	// Primary errors before producing any chunk.
	f.srv.deps.Providers["primary"] = &fake.Err{NameStr: "primary", Kind: types.ErrServerError}

	w := doRequest(f.srv, map[string]any{
		"model":    "auto",
		"stream":   true,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	body, _ := io.ReadAll(w.Body)
	if !strings.Contains(string(body), "data:") || !strings.Contains(string(body), "[DONE]") {
		t.Errorf("expected SSE body from fallback, got: %q", body)
	}
	if f.fallback.Calls.Load() != 1 {
		t.Errorf("fallback should have served the stream, calls=%d", f.fallback.Calls.Load())
	}
}

func TestHandlerStreamMidwayFailureNoFallback(t *testing.T) {
	f := newFixture(t)
	mid := &fake.StreamFailMidway{NameStr: "primary", Chunks: 2}
	f.srv.deps.Providers["primary"] = mid

	w := doRequest(f.srv, map[string]any{
		"model":    "auto",
		"stream":   true,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d", w.Code)
	}
	body, _ := io.ReadAll(w.Body)
	// Mid-stream failure should NOT trigger fallback — chunks already
	// went to the client.
	if f.fallback.Calls.Load() != 0 {
		t.Errorf("mid-stream failure must not fall back, fallback calls=%d", f.fallback.Calls.Load())
	}
	if !strings.Contains(string(body), "event: error") {
		t.Errorf("expected SSE error event in body, got: %q", body)
	}
}

func TestHandlerPromptCacheHit(t *testing.T) {
	f := newFixture(t)
	// First call populates cache.
	doRequest(f.srv, map[string]any{
		"model":    "auto",
		"messages": []map[string]any{{"role": "user", "content": "cache me"}},
	})
	calls := f.primary.Calls.Load()

	w := doRequest(f.srv, map[string]any{
		"model":    "auto",
		"messages": []map[string]any{{"role": "user", "content": "cache me"}},
	})
	if w.Header().Get("X-Nadir-Cache") != "hit" {
		t.Errorf("expected cache hit header, got %q", w.Header().Get("X-Nadir-Cache"))
	}
	if got := f.primary.Calls.Load(); got != calls {
		t.Errorf("cached request should not invoke primary again, calls=%d (was %d)", got, calls)
	}
}
