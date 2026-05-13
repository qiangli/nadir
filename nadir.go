// Package nadir is an OpenAI-compatible LLM router with embedding-based
// prompt classification. It exposes the routing decision pipeline used
// by the `nadir` binary as a library: tier a prompt with a small
// embedding model (or a heuristic), pick the cheapest model that can
// handle it, and fall back through a chain on transient errors.
//
// # Quick start
//
// The simplest path is to build a Classifier and call it directly:
//
//	import "github.com/qiangli/nadir/classifier"
//
//	cls := classifier.NewHeuristic(classifier.DefaultThresholds())
//	tier, score, conf, err := cls.Classify(ctx, "Refactor this concurrent code")
//	// tier = "complex", score = 0.85, conf = 0.7
//
// For production-grade classification, build with `-tags onnx` and
// load the MiniLM ONNX model (see tools/export-onnx/):
//
//	cls, err := classifier.NewONNXFromAssets("./assets", classifier.DefaultThresholds())
//
// For routing decisions (classify + alias + modifiers + fallback chain),
// import the router package:
//
//	import "github.com/qiangli/nadir/router"
//
//	r := router.New(cfg, cls, sess, healthTracker)
//	decision, err := r.Route(ctx, req, user)
//	// decision.Model, decision.Tier, decision.FallbackChain
//
// # Package layout
//
//   - types     — wire and internal types (ChatRequest, Tier, RouteDecision, …)
//   - classifier — Classifier interface plus Heuristic, ONNX, LLM, and Cascading impls
//   - embed     — pure-Go BERT tokenizer plus an ONNX session wrapper (-tags onnx)
//   - router    — smart routing: classify → alias → modifiers → fallback chain
//   - cache     — PromptCache (LRU+TTL) and SessionCache (upgrade-only)
//   - ratelimit — per-user and per-model sliding-window rate limiters
//   - health    — per-model failure tracking + cooldown
//   - modelmeta — embedded model cost/context/vision lookup table
//   - provider/openai — OpenAI-compatible HTTP client (also covers Ollama, vLLM, LocalAI via base-URL)
//   - provider/fake   — test doubles
//
// The HTTP proxy server, persistence (SQLite/JSONL), Prometheus
// metrics, and CLI tools live under internal/ since they're
// opinionated app glue rather than reusable library surface.
//
// # Build tags
//
//   - default build — heuristic classifier only. No CGO, no ML deps.
//   - -tags onnx    — adds the ONNX-backed MiniLM classifier. Requires
//     a C compiler at build time and libonnxruntime
//     (loaded via dlopen) at runtime.
package nadir

import (
	"github.com/qiangli/nadir/classifier"
	"github.com/qiangli/nadir/router"
	"github.com/qiangli/nadir/skillrouter"
	"github.com/qiangli/nadir/types"
)

// Re-exports of the most commonly used types so callers can write
// `nadir.Tier` and `nadir.RouteDecision` without importing the
// sub-packages directly. For deeper access, import the sub-packages.
type (
	Tier          = types.Tier
	Message       = types.Message
	ChatRequest   = types.ChatRequest
	ChatResponse  = types.ChatResponse
	RouteDecision = types.RouteDecision
	User          = types.User
	Classifier    = types.Classifier
	Router        = types.Router
	LLMClient     = types.LLMClient
	ProviderError = types.ProviderError

	// Skill-routing surface (parallel to model-tier routing).
	Skill         = skillrouter.Skill
	SkillRouter   = skillrouter.Router
	SkillDecision = skillrouter.Decision
	SkillMatcher  = skillrouter.Matcher
	SkillSemantic = skillrouter.Semantic
	SkillCascade  = skillrouter.Cascade
	SkillEmbedder = skillrouter.Embedder
	SkillRanked   = skillrouter.Ranked
)

// Tier constants re-exported for convenience.
const (
	TierSimple    = types.TierSimple
	TierMid       = types.TierMid
	TierComplex   = types.TierComplex
	TierReasoning = types.TierReasoning
	TierFree      = types.TierFree
)

// NewHeuristicClassifier returns a Classifier that scores prompts
// using length + code-density + keyword signals. Zero deps, runs in
// microseconds; suitable for development, tests, and as a no-ONNX
// fallback. For production routing, use NewONNXClassifier (requires
// `-tags onnx` build + generated assets).
func NewHeuristicClassifier() Classifier {
	return classifier.NewHeuristic(classifier.DefaultThresholds())
}

// IsTransientError reports whether an error from an LLMClient is
// a transient kind (rate-limit, 5xx, timeout, network). The router's
// fallback loop uses this to decide between retry-next-model and
// abort-surface-to-caller.
func IsTransientError(err error) bool {
	return types.IsTransient(err)
}

// NewSkillRouter is a convenience constructor for the skill-routing
// pipeline. It picks the best slash-command / tool name for a prompt
// from a caller-registered catalog by asking a small LLM (typically
// Ollama llama3.2:3b). For options (timeout, logger, max tokens),
// import skillrouter directly.
func NewSkillRouter(client LLMClient, model string, skills []Skill) *SkillRouter {
	return skillrouter.New(client, model, skills)
}

// _ packages referenced for godoc cross-linking only.
var (
	_ = classifier.DefaultThresholds
	_ = router.New
	_ = skillrouter.New
)
