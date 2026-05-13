package skillrouter

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/qiangli/nadir/types"
)

// Cascade is the hybrid two-stage matcher the research converges on:
// a fast embedding primary handles the easy 80%, an LLM secondary
// reranks only the uncertain shortlist. Production reports cluster
// around "embedding-first with LLM fallback" delivering ~94% accuracy
// at ~60% of LLM-only cost — see REIC, Semantic Router patterns.
//
// Trigger logic — Cascade calls the LLM when *either* signal goes
// soft, not just absolute score:
//
//   - Top-1 score < MinScore           → embeddings can't decide at all
//   - Top1-Top2 Margin < MinMargin     → embeddings are split between
//     close competitors (the /review vs /security-review case)
//
// On both, only the top-K candidates are handed to the LLM (RAG-style
// shortlist), keeping the prompt short and sidestepping the
// position-bias problem that scales with catalog size.
//
// Failure semantics mirror classifier.Cascading: if the LLM errors or
// times out, fall back to the primary's verdict silently. Routing
// must always produce an answer; the worst case is a single sub-optimal
// route, not a 500.
type Cascade struct {
	primary    *Semantic
	llmClient  types.LLMClient
	llmModel   string
	minMargin  float64
	shortlistK int
	llmTimeout time.Duration
	maxTokens  int
	logger     *slog.Logger
}

// CascadeOption configures a Cascade matcher.
type CascadeOption func(*Cascade)

// WithMinMargin sets the top1-top2 gap below which the LLM is
// consulted even when top-1 score is high. Default 0.10 — a typical
// "two close competitors" signal in L2-normalized MiniLM space.
func WithMinMargin(m float64) CascadeOption {
	return func(c *Cascade) { c.minMargin = m }
}

// WithShortlistK caps how many top-ranked candidates are passed to
// the LLM. Default 5 — keeps the rerank prompt short while still
// allowing the LLM to override a close primary call. Setting K to a
// value larger than the catalog is harmless (treated as the full
// catalog).
func WithShortlistK(k int) CascadeOption {
	return func(c *Cascade) {
		if k > 0 {
			c.shortlistK = k
		}
	}
}

// WithLLMTimeout caps the per-call LLM rerank deadline. Default 2s,
// matching classifier.Cascading. The LLM is the slow path; when it
// hangs, the primary verdict must still ship promptly.
func WithLLMTimeout(d time.Duration) CascadeOption {
	return func(c *Cascade) { c.llmTimeout = d }
}

// WithCascadeLogger sets the slog.Logger.
func WithCascadeLogger(l *slog.Logger) CascadeOption {
	return func(c *Cascade) {
		if l != nil {
			c.logger = l
		}
	}
}

// NewCascade builds a Cascade matcher around a fully-constructed
// Semantic primary and an LLMClient for the rerank step. The LLM
// model name and any rerank-tuning options use the same env-var
// defaults documented for Router (NADIR_CASCADE_LLM_MODEL, etc.).
func NewCascade(primary *Semantic, llmClient types.LLMClient, llmModel string, opts ...CascadeOption) *Cascade {
	c := &Cascade{
		primary:    primary,
		llmClient:  llmClient,
		llmModel:   llmModel,
		minMargin:  0.10,
		shortlistK: 5,
		llmTimeout: 2 * time.Second,
		maxTokens:  32,
		logger:     slog.Default(),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Route picks a skill. Primary embedding always runs (it's cheap);
// the LLM secondary fires only when the embedding signal is weak.
func (c *Cascade) Route(ctx context.Context, prompt string) (*Decision, error) {
	if c.primary == nil {
		return nil, errEmptyPrimary
	}

	// Always need the ranked candidates, both to make the primary
	// verdict and to build the shortlist for the LLM if we escalate.
	ranked, err := c.primary.Rank(ctx, prompt, c.shortlistK)
	if err != nil {
		return nil, err
	}
	if len(ranked) == 0 {
		return &Decision{FellThrough: true}, nil
	}

	top := ranked[0]
	margin := 0.0
	if len(ranked) > 1 {
		margin = top.Score - ranked[1].Score
	}

	// Below MinScore: embeddings can't decide. Skip the LLM — there's
	// nothing useful to rerank if no catalog entry is even close.
	if top.Score < c.primary.minScore {
		c.logger.Debug("cascade: fellthrough (no candidate above minScore)",
			slog.Float64("top_score", top.Score))
		return &Decision{FellThrough: true, Confidence: top.Score, Margin: margin}, nil
	}

	// Above MinScore AND comfortable margin: trust the primary.
	if margin >= c.minMargin || c.llmClient == nil || c.llmModel == "" {
		return &Decision{Skill: top.Skill, Confidence: top.Score, Margin: margin}, nil
	}

	// Margin too narrow: ask the LLM, but only about the shortlist.
	shortlist := c.makeShortlist(ranked)
	c.logger.Debug("cascade: consulting LLM",
		slog.Float64("top_score", top.Score),
		slog.Float64("margin", margin),
		slog.Int("shortlist_size", len(shortlist)),
	)
	d, err := c.rerank(ctx, prompt, shortlist)
	if err != nil {
		c.logger.Warn("cascade: LLM rerank failed; using primary verdict",
			slog.Float64("top_score", top.Score),
			slog.Float64("margin", margin),
			slog.Any("err", err),
		)
		return &Decision{Skill: top.Skill, Confidence: top.Score, Margin: margin}, nil
	}
	// Preserve the embedding-side margin so callers can see how
	// uncertain the primary was even when the LLM made the final call.
	if !d.FellThrough && d.Margin == 0 {
		d.Margin = margin
	}
	return d, nil
}

// makeShortlist resolves the top-K Ranked entries back into the
// caller-registered Skill structs (Name + Description) so the LLM
// rerank prompt sees descriptions, not just names. Order is preserved
// (highest similarity first) — even though Router internally renders
// the catalog as a flat list, the ranking is information the LLM can
// implicitly anchor on.
func (c *Cascade) makeShortlist(ranked []Ranked) []Skill {
	byName := make(map[string]Skill, len(c.primary.skills))
	for _, s := range c.primary.skills {
		byName[s.Name] = s
	}
	out := make([]Skill, 0, len(ranked))
	for _, r := range ranked {
		if s, ok := byName[r.Skill]; ok {
			out = append(out, s)
		}
	}
	return out
}

// rerank runs an ephemeral LLM Router constructed with only the
// shortlist. This is cheaper than holding a long-lived rerank Router
// because the catalog changes per call — and constructing a Router is
// just struct allocation.
func (c *Cascade) rerank(ctx context.Context, prompt string, shortlist []Skill) (*Decision, error) {
	r := New(c.llmClient, c.llmModel, shortlist,
		WithTimeout(c.llmTimeout),
		WithMaxTokens(c.maxTokens),
		WithLogger(c.logger),
	)
	return r.Route(ctx, prompt)
}

var errEmptyPrimary = errors.New("skillrouter: cascade has no primary")

var _ Matcher = (*Cascade)(nil)
