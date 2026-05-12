// Package proxy is the HTTP-facing layer: chi router, middleware
// chain, and the /v1/chat/completions handler that ties classifier +
// router + LLM client + fallback loop together.
//
// The fallback semantics differ between batch and streaming responses:
// batch calls cascade through the chain transparently (the client
// sees one final response or error), while streaming calls can only
// fall back until the first chunk goes out — once the SSE framing is
// committed, mid-stream errors surface as an SSE `error` event.
package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"time"

	"github.com/go-chi/chi/v5"
	chimw "github.com/go-chi/chi/v5/middleware"

	"github.com/qiangli/nadir/internal/metrics"
	"github.com/qiangli/nadir/types"
)

type Deps struct {
	Logger          *slog.Logger
	Router          types.Router
	Classifier      types.Classifier
	Providers       map[string]types.LLMClient // keyed by provider name
	PromptCache     types.PromptCache
	UserLimiter     types.UserRateLimiter
	ModelLimiter    types.ModelRateLimiter
	Health          types.ProviderHealth
	ModelToProvider func(model string) string
	AuthToken       string
	PerCallTimeout  time.Duration
	MaxBodyBytes    int64

	// Phase 2 wiring (all optional — proxy works without them).
	Metrics *metrics.Collectors
	Loggers []types.RequestLogger
	Budget  types.Budget

	// ClassifierLabel is surfaced in /health for monitoring so
	// dashboards can alert when a deploy flips to a degraded mode.
	ClassifierLabel string
}

type Server struct {
	deps   Deps
	router chi.Router
	pool   *loggerPool
}

func New(d Deps) *Server {
	if d.Logger == nil {
		d.Logger = slog.Default()
	}
	if d.PerCallTimeout == 0 {
		d.PerCallTimeout = 2 * time.Minute
	}
	if d.MaxBodyBytes == 0 {
		d.MaxBodyBytes = 20 * 1024 * 1024
	}
	s := &Server{deps: d}
	if len(d.Loggers) > 0 {
		s.pool = newLoggerPool(d.Logger, d.Loggers, 4, 1024)
	}
	s.router = s.buildRouter()
	return s
}

// Close shuts down the logger pool. Safe to call multiple times.
func (s *Server) Close() {
	if s.pool != nil {
		s.pool.Close()
	}
}

func (s *Server) Handler() http.Handler { return s.router }

func (s *Server) buildRouter() chi.Router {
	r := chi.NewRouter()
	r.Use(chimw.Recoverer)
	r.Use(chimw.RequestID)
	r.Use(chimw.RealIP)
	r.Use(s.accessLog)
	r.Use(s.bodySizeLimit)
	r.Use(s.authMiddleware)
	r.Use(s.userRateLimit)

	r.Get("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		label := s.deps.ClassifierLabel
		if label == "" {
			label = "unknown"
		}
		fmt.Fprintf(w, `{"status":"ok","classifier":%q}`, label)
	})
	if s.deps.Metrics != nil {
		r.Method(http.MethodGet, "/metrics", s.deps.Metrics.Handler())
	}
	r.Post("/v1/chat/completions", s.handleChatCompletions)
	r.Post("/v1/classify", s.handleClassify)
	r.Get("/v1/models", s.handleModels)
	r.Get("/v1/cache", s.handleCacheStats)
	r.Get("/v1/budget", s.handleBudget)

	return r
}

func (s *Server) handleCacheStats(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if s.deps.PromptCache == nil {
		fmt.Fprint(w, `{}`)
		return
	}
	_ = json.NewEncoder(w).Encode(s.deps.PromptCache.Stats())
}

func (s *Server) handleBudget(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if s.deps.Budget == nil {
		fmt.Fprint(w, `{}`)
		return
	}
	// Budget interface only exposes Allowed/Estimate; concrete tracker
	// has Snapshot. Use type assertion to expose richer state when
	// available.
	type snapshotter interface {
		Snapshot() any
	}
	if sn, ok := s.deps.Budget.(interface{ Snapshot() any }); ok {
		_ = json.NewEncoder(w).Encode(sn.Snapshot())
		return
	}
	allowed, reason := s.deps.Budget.Allowed()
	_ = json.NewEncoder(w).Encode(map[string]any{"allowed": allowed, "reason": reason})
}

func (s *Server) accessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := chimw.NewWrapResponseWriter(w, r.ProtoMajor)
		next.ServeHTTP(rw, r)
		s.deps.Logger.Info("request",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", rw.Status()),
			slog.Duration("latency", time.Since(start)),
		)
	})
}

func (s *Server) bodySizeLimit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Body != nil {
			r.Body = http.MaxBytesReader(w, r.Body, s.deps.MaxBodyBytes)
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.deps.AuthToken == "" {
			next.ServeHTTP(w, r)
			return
		}
		// /health bypasses auth.
		if r.URL.Path == "/health" {
			next.ServeHTTP(w, r)
			return
		}
		got := r.Header.Get("Authorization")
		want := "Bearer " + s.deps.AuthToken
		if got != want {
			writeError(w, http.StatusUnauthorized, "missing or invalid bearer token")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) userRateLimit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.deps.UserLimiter == nil {
			next.ServeHTTP(w, r)
			return
		}
		userID := userIDFromRequest(r)
		if retry, ok := s.deps.UserLimiter.Check(userID); !ok {
			w.Header().Set("Retry-After", strconv.Itoa(int(retry.Seconds())+1))
			writeError(w, http.StatusTooManyRequests, "user rate limit exceeded")
			return
		}
		next.ServeHTTP(w, r)
	})
}

func userIDFromRequest(r *http.Request) string {
	if u, _, ok := r.BasicAuth(); ok && u != "" {
		return u
	}
	if h := r.Header.Get("X-User-ID"); h != "" {
		return h
	}
	return r.RemoteAddr
}

func (s *Server) handleClassify(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Prompt string `json:"prompt"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	tier, score, conf, err := s.deps.Classifier.Classify(r.Context(), body.Prompt)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "classify: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"tier":       tier,
		"score":      score,
		"confidence": conf,
	})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	out := map[string]any{
		"object": "list",
		"data":   []any{},
	}
	_ = json.NewEncoder(w).Encode(out)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req types.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages must be non-empty")
		return
	}

	user := &types.User{ID: userIDFromRequest(r)}

	// Budget gate (before route, before upstream call).
	if s.deps.Budget != nil {
		if ok, reason := s.deps.Budget.Allowed(); !ok {
			writeError(w, http.StatusPaymentRequired, "budget: "+reason)
			return
		}
	}

	if !req.Stream && s.deps.PromptCache != nil {
		if cached, ok := s.deps.PromptCache.Get(req.Model, req.Messages); ok {
			writeRoutingHeaders(w, &types.RouteDecision{Model: cached.Model, Tier: types.TierSimple, Cached: true})
			w.Header().Set("X-Nadir-Cache", "hit")
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(cached)
			if s.deps.Metrics != nil {
				s.deps.Metrics.CacheHits.Inc()
				s.deps.Metrics.Observe(cached.Model, "cached", "ok", time.Since(start), 0, 0, 0)
			}
			return
		}
		if s.deps.Metrics != nil {
			s.deps.Metrics.CacheMisses.Inc()
		}
	}

	decision, err := s.deps.Router.Route(r.Context(), &req, user)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "route: "+err.Error())
		return
	}

	chain := append([]string{decision.Model}, decision.FallbackChain...)
	writeRoutingHeaders(w, decision)

	if req.Stream {
		s.streamWithFallback(r.Context(), w, &req, decision, chain)
		s.recordRequest(user, &req, decision, "ok", time.Since(start), nil)
		return
	}

	resp, err := s.completeWithFallback(r.Context(), &req, decision, chain)
	if err != nil {
		var pe *types.ProviderError
		status := http.StatusBadGateway
		if errors.As(err, &pe) {
			switch pe.Kind {
			case types.ErrAuth:
				status = http.StatusUnauthorized
			case types.ErrBadRequest, types.ErrValidation:
				status = http.StatusBadRequest
			case types.ErrRateLimit:
				status = http.StatusTooManyRequests
			}
		}
		writeError(w, status, err.Error())
		s.recordRequest(user, &req, decision, "error", time.Since(start), nil)
		return
	}

	if s.deps.PromptCache != nil {
		s.deps.PromptCache.Put(req.Model, req.Messages, resp)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
	s.recordRequest(user, &req, decision, "ok", time.Since(start), resp)
}

// recordRequest fires the post-response observability fan-out: emit
// Prometheus counters/histograms, hand the entry to the logger pool
// (which persists to SQLite + JSONL asynchronously), and update the
// budget tracker. Failures here never affect the response.
func (s *Server) recordRequest(user *types.User, req *types.ChatRequest, decision *types.RouteDecision, status string, latency time.Duration, resp *types.ChatResponse) {
	var promptTok, completionTok int
	if resp != nil && resp.Usage != nil {
		promptTok = resp.Usage.PromptTokens
		completionTok = resp.Usage.CompletionTokens
	}
	cost := 0.0
	if s.deps.Budget != nil {
		cost = s.deps.Budget.Estimate(decision.Model, promptTok, completionTok)
		if cost > 0 {
			s.deps.Budget.Record(decision.Model, cost)
		}
	}
	if s.deps.Metrics != nil {
		s.deps.Metrics.Observe(decision.Model, string(decision.Tier), status, latency, promptTok, completionTok, cost)
	}
	if s.pool != nil {
		entry := &types.RequestEntry{
			ID:               chimwRequestID(req),
			Timestamp:        time.Now(),
			Model:            decision.Model,
			Tier:             decision.Tier,
			Provider:         decision.Provider,
			PromptTokens:     promptTok,
			CompletionTokens: completionTok,
			CostUSD:          cost,
			LatencyMs:        latency.Milliseconds(),
			Status:           status,
			UserID:           user.ID,
			Score:            decision.Score,
			Confidence:       decision.Confidence,
			Modifiers:        decision.Modifiers,
		}
		s.pool.Submit(entry)
	}
}

// chimwRequestID derives an ID from the inflight ChatRequest body —
// not the chi RequestID middleware value, since we don't have access
// to the request context here. A timestamp-based ID is fine for
// SQLite uniqueness given the worker pool is single-writer-per-row.
func chimwRequestID(req *types.ChatRequest) string {
	_ = req
	return fmt.Sprintf("req-%d", time.Now().UnixNano())
}

func writeRoutingHeaders(w http.ResponseWriter, d *types.RouteDecision) {
	w.Header().Set("X-Routed-Model", d.Model)
	w.Header().Set("X-Routed-Tier", string(d.Tier))
	if d.Score > 0 {
		w.Header().Set("X-Complexity-Score", fmt.Sprintf("%.4f", d.Score))
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
			"code":    code,
		},
	})
}

// pickProvider resolves a model name to an LLMClient using the
// ModelToProvider mapping or falling back to the "openai" client. If
// no client is configured it returns an Auth error so the fallback
// loop treats the situation as fatal (no point retrying).
func (s *Server) pickProvider(model string) (types.LLMClient, error) {
	name := "openai"
	if s.deps.ModelToProvider != nil {
		if got := s.deps.ModelToProvider(model); got != "" {
			name = got
		}
	}
	client, ok := s.deps.Providers[name]
	if !ok {
		return nil, &types.ProviderError{
			Kind:     types.ErrAuth,
			Provider: name,
			Model:    model,
			Msg:      fmt.Sprintf("no client configured for provider %q", name),
		}
	}
	return client, nil
}
