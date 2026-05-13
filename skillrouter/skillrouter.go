// Package skillrouter picks the best skill (slash-command, tool name,
// or named capability) for a user prompt by asking a small LLM to
// choose from a caller-registered catalog.
//
// It is a sibling pipeline to package router: same OpenAI-compatible
// LLMClient plumbing, same defensive-parsing posture as classifier.LLM,
// but the output is a skill name from a fixed catalog instead of a
// model-tier label. Tier ranks, fallback chains, and modifier
// upgrades from package router are intentionally not reused — skill
// dispatch is one-shot and orthogonal to model selection.
//
// # Quick start
//
//	skills := []skillrouter.Skill{
//	    {Name: "/review", Description: "Review a pull request"},
//	    {Name: "/security-review", Description: "Audit pending changes for security issues"},
//	    {Name: "/init", Description: "Initialize a CLAUDE.md for this repo"},
//	}
//	client := openai.New("ollama", "http://localhost:11434/v1", "")
//	r := skillrouter.New(client, "llama3.2:3b", skills)
//	decision, err := r.Route(ctx, "audit this branch for vulnerabilities")
//	// decision.Skill == "/security-review"
//
// When no skill in the catalog applies, Route returns
// Decision{FellThrough: true} so the caller can fall back to a default
// handler. The LLM is instructed to reply with "none" in that case,
// and any unparseable reply is treated as a fallthrough — never as a
// wrong skill pick.
package skillrouter

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/qiangli/nadir/types"
)

// Skill is one entry in a caller-registered catalog. Name is the
// canonical identifier the router will return (typically a
// slash-command like "/review"); Description is a one-line summary
// shown to the LLM in LLM-backed matchers; Examples are seed prompts
// that the Semantic matcher embeds as prototype vectors (one cosine
// score per example, max wins) — the multi-vector prototypes trick
// that drives the +15% on CLINC150 over single-centroid retrieval.
// Trigger phrases ("audit for vulnerabilities") belong in Examples;
// the embedder generalizes them across paraphrases the way a regex
// can't.
type Skill struct {
	Name        string
	Description string
	Examples    []string
}

// Decision is the router's output. Skill is the chosen catalog entry's
// Name (empty when FellThrough). Confidence is in [0,1]; semantics
// depend on the backend:
//
//   - Router (LLM): parser-derived — 1.0 exact name, 0.9 case-insensitive,
//     0.7 substring match.
//   - Semantic: top-1 cosine similarity to any prototype vector.
//   - Cascade: passes through the deciding matcher's score.
//
// Margin is the gap between the top-1 and top-2 candidate scores. It
// matters for embedding-based backends: a top-1 of 0.78 with a
// runner-up of 0.77 is genuinely uncertain even though 0.78 is "high"
// — Cascade uses Margin (not Confidence alone) to decide whether to
// consult its secondary. LLM-backed matchers leave Margin = 0.
//
// RawResponse is the LLM's verbatim reply when an LLM was consulted,
// useful for logging and debugging.
//
// FellThrough=true means no skill applied — the caller should run its
// default handler.
type Decision struct {
	Skill       string
	Confidence  float64
	Margin      float64
	RawResponse string
	FellThrough bool
}

// Matcher is the common interface for the three skill-matching
// strategies in this package: Router (LLM-only), Semantic
// (embedding-only), and Cascade (Semantic primary + LLM rerank on
// shortlist). Mirrors classifier.Classifier in shape so the cascade
// composes the same way classifier.Cascading does.
type Matcher interface {
	Route(ctx context.Context, prompt string) (*Decision, error)
}

// Router picks a skill for a prompt by asking an upstream LLM. Concurrency-safe.
type Router struct {
	client    types.LLMClient
	model     string
	skills    []Skill
	timeout   time.Duration
	maxTokens int
	logger    *slog.Logger
}

// Option configures a Router.
type Option func(*Router)

// WithTimeout caps the per-call upstream LLM timeout. Zero (default)
// means use the parent ctx's deadline only.
func WithTimeout(d time.Duration) Option {
	return func(r *Router) { r.timeout = d }
}

// WithLogger sets the slog.Logger used for warning-level diagnostics
// (parse failures, timeouts). Defaults to slog.Default().
func WithLogger(l *slog.Logger) Option {
	return func(r *Router) {
		if l != nil {
			r.logger = l
		}
	}
}

// WithMaxTokens caps the upstream completion size. Defaults to 32,
// which is enough room for a skill name (e.g., "/security-review")
// plus a stray space or newline.
func WithMaxTokens(n int) Option {
	return func(r *Router) {
		if n > 0 {
			r.maxTokens = n
		}
	}
}

// New builds a Router. client and at least one skill are required;
// model defaults to "llama3.2:3b" to match the cascading-classifier
// default. Pass options to override the timeout, logger, or max tokens.
func New(client types.LLMClient, model string, skills []Skill, opts ...Option) *Router {
	if model == "" {
		model = "llama3.2:3b"
	}
	r := &Router{
		client:    client,
		model:     model,
		skills:    skills,
		maxTokens: 32,
		logger:    slog.Default(),
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// Route asks the LLM to pick a skill for the prompt and returns the
// matched Decision. An error is returned only for transport-level
// problems (nil client, empty catalog, upstream error after timeout).
// A reply that doesn't match any catalog entry is not an error — it's
// a Decision with FellThrough=true.
func (r *Router) Route(ctx context.Context, prompt string) (*Decision, error) {
	if r.client == nil {
		return nil, errors.New("skillrouter: nil client")
	}
	if len(r.skills) == 0 {
		return nil, errors.New("skillrouter: empty skill catalog")
	}

	if r.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.timeout)
		defer cancel()
	}

	zero := 0.0
	req := &types.ChatRequest{
		Model: r.model,
		Messages: []types.Message{
			{Role: "user", Content: jsonString(renderPrompt(r.skills, prompt))},
		},
		MaxTokens:   &r.maxTokens,
		Temperature: &zero, // routing must be deterministic; same prompt → same skill
	}
	resp, err := r.client.Complete(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("skillrouter: %w", err)
	}
	if resp == nil || len(resp.Choices) == 0 {
		return nil, errors.New("skillrouter: empty response")
	}

	raw := extractText(resp.Choices[0].Message.Content)
	d := parseSkill(raw, r.skills)
	d.RawResponse = raw
	if d.FellThrough {
		r.logger.Debug("skillrouter: fellthrough", slog.String("raw", raw))
	}
	return d, nil
}

var _ Matcher = (*Router)(nil)
