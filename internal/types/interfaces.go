package types

import (
	"context"
	"io"
	"time"
)

// LLMClient is the upstream-provider abstraction. Every provider
// (OpenAI, Anthropic, Gemini, Ollama, LiteLLM long tail, plus the
// fakes used in tests) implements this interface. Complete returns a
// single response; Stream returns an iterator that yields chunks until
// the underlying connection closes or the upstream sends the [DONE]
// sentinel.
type LLMClient interface {
	Name() string
	Complete(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
	Stream(ctx context.Context, req *ChatRequest) (StreamIter, error)
}

// StreamIter yields stream chunks one at a time. Next returns io.EOF
// when the stream is exhausted. The final chunk may carry Usage data.
type StreamIter interface {
	Next(ctx context.Context) (*StreamChunk, error)
	io.Closer
}

// Classifier maps a prompt to a complexity tier. Score is in [0,1]
// where higher = more complex; Confidence is the absolute difference
// between the simple and complex centroid similarities.
type Classifier interface {
	Classify(ctx context.Context, prompt string) (Tier, float64, float64, error)
	Warmup(ctx context.Context) error
}

// Router resolves a ChatRequest into a RouteDecision: which model to
// call, the fallback chain, the classifier score, and any modifiers
// applied (agentic, reasoning, vision overrides, etc.).
type Router interface {
	Route(ctx context.Context, req *ChatRequest, user *User) (*RouteDecision, error)
}

// PromptCache is the request-level cache: identical model + messages
// hash → cached response. TTL+LRU bounded.
type PromptCache interface {
	Get(model string, msgs []Message) (*ChatResponse, bool)
	Put(model string, msgs []Message, resp *ChatResponse)
	Stats() CacheStats
}

type CacheStats struct {
	Size     int
	Hits     uint64
	Misses   uint64
	Evictions uint64
}

// SessionCache pins the model picked for a (system + first user
// message) pair so multi-turn conversations don't bounce between tiers.
// UpgradeIfHigher returns the recorded model if cached, else stores the
// caller's pick. It never downgrades.
type SessionCache interface {
	Get(msgs []Message) (model string, tier Tier, ok bool)
	UpgradeIfHigher(msgs []Message, model string, tier Tier) (finalModel string, finalTier Tier, status string)
}

// UserRateLimiter throttles per caller (or per anonymous IP).
type UserRateLimiter interface {
	Check(userID string) (retryAfter time.Duration, ok bool)
}

// ModelRateLimiter throttles per upstream model so a 429 storm against
// one provider doesn't starve the others.
type ModelRateLimiter interface {
	Check(model string) (retryAfter time.Duration, ok bool)
	Record(model string, retryAfter time.Duration)
}

// ProviderHealth records success/failure per model and reorders a
// candidate chain so unhealthy models drift to the back.
type ProviderHealth interface {
	RecordSuccess(model string)
	RecordFailure(model string, kind ErrorKind)
	Order(candidates []string) []string
	Available(model string) bool
}

// RequestLogger persists one row per request to SQLite (Phase 2). The
// Phase 1 binary uses a no-op implementation.
type RequestLogger interface {
	Log(ctx context.Context, entry *RequestEntry) error
}

type RequestEntry struct {
	ID               string    `json:"id"`
	Timestamp        time.Time `json:"ts"`
	Model            string    `json:"model"`
	Tier             Tier      `json:"tier"`
	Provider         string    `json:"provider"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	CostUSD          float64   `json:"cost_usd"`
	LatencyMs        int64     `json:"latency_ms"`
	Status           string    `json:"status"`
	UserID           string    `json:"user_id,omitempty"`
	Score            float64   `json:"score,omitempty"`
	Confidence       float64   `json:"confidence,omitempty"`
	Modifiers        []string  `json:"modifiers,omitempty"`
}

// Budget gates requests against daily/monthly spending limits.
type Budget interface {
	Allowed() (allowed bool, reason string)
	Record(model string, cost float64)
	Estimate(model string, promptTokens, completionTokens int) float64
}

// TokenProvider resolves a provider name to an auth token. The Phase 1
// implementation is a static lookup against credentials.json + env vars;
// Phase 5 will plug in real OAuth refresh.
type TokenProvider interface {
	Token(ctx context.Context, provider string) (string, error)
}
