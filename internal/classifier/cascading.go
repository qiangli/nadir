package classifier

// Cascading is a defense-in-depth Classifier: a fast Primary
// (heuristic or ONNX/MiniLM) makes the first call; if its confidence
// is below ConfidenceThreshold, a slower Secondary (typically an
// LLM-backed classifier) is consulted for a second opinion.
//
// This is NadirClaw's idea — when you're uncertain, ask a model — but
// expressed as a decorator so it composes with any Classifier
// implementation without bloating the primary.
//
// Semantics: if Secondary errors or times out, fall back to Primary's
// original (low-confidence) verdict rather than failing the request.
// Routing must always produce an answer; the worst case is a single
// misrouted prompt, not a 500.

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/qiangli/nadir/internal/types"
)

type Cascading struct {
	Primary             types.Classifier
	Secondary           types.Classifier
	ConfidenceThreshold float64
	SecondaryTimeout    time.Duration
	Logger              *slog.Logger
}

// NewCascading wires Primary and Secondary. ConfidenceThreshold is
// the cutoff below which we consult Secondary: 0.0 means "always
// trust Primary"; 1.0 means "always consult Secondary". A typical
// production value is 0.3–0.5 (consult on genuinely ambiguous
// prompts, trust Primary on clear ones).
func NewCascading(primary, secondary types.Classifier, threshold float64) *Cascading {
	return &Cascading{
		Primary:             primary,
		Secondary:           secondary,
		ConfidenceThreshold: threshold,
		SecondaryTimeout:    2 * time.Second,
		Logger:              slog.Default(),
	}
}

func (c *Cascading) Warmup(ctx context.Context) error {
	if err := c.Primary.Warmup(ctx); err != nil {
		return err
	}
	if c.Secondary != nil {
		// Best-effort: a Secondary warmup failure (e.g. Ollama not
		// running yet) shouldn't break startup. We log and continue.
		if err := c.Secondary.Warmup(ctx); err != nil && c.Logger != nil {
			c.Logger.Warn("cascade secondary warmup failed", slog.Any("err", err))
		}
	}
	return nil
}

func (c *Cascading) Classify(ctx context.Context, prompt string) (types.Tier, float64, float64, error) {
	tier, score, conf, err := c.Primary.Classify(ctx, prompt)
	if err != nil {
		return tier, score, conf, err
	}
	if c.Secondary == nil || conf >= c.ConfidenceThreshold {
		return tier, score, conf, nil
	}

	// Confidence below threshold — consult Secondary with a bounded budget.
	secCtx, cancel := context.WithTimeout(ctx, c.SecondaryTimeout)
	defer cancel()
	tier2, score2, conf2, err2 := c.Secondary.Classify(secCtx, prompt)
	if err2 != nil {
		if c.Logger != nil {
			c.Logger.Warn("cascade secondary failed; using primary verdict",
				slog.Float64("primary_score", score),
				slog.Float64("primary_confidence", conf),
				slog.Any("err", err2),
			)
		}
		return tier, score, conf, nil
	}
	if c.Logger != nil {
		c.Logger.Debug("cascade consulted secondary",
			slog.Float64("primary_score", score),
			slog.Float64("primary_confidence", conf),
			slog.Float64("secondary_score", score2),
			slog.String("primary_tier", string(tier)),
			slog.String("secondary_tier", string(tier2)),
		)
	}
	return tier2, score2, conf2, nil
}

// Compile-time interface check.
var _ types.Classifier = (*Cascading)(nil)

// errCheck keeps the package honest about unused imports during
// development. Removed once the package is referenced from main.
var _ = errors.New
