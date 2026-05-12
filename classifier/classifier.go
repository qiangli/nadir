// Package classifier maps a prompt to a complexity tier. The
// production implementation will pool a MiniLM ONNX embedding against
// two pre-computed centroids; the v1 build ships a heuristic
// implementation (length + keyword scoring) so the proxy hot path can
// be exercised before the ONNX export tooling lands.
//
// Numerics match priorart/NadirClaw/nadirclaw/classifier.py once the
// ONNX backend is wired in: same dot-product → confidence → score →
// tier thresholding pipeline.
package classifier

import (
	"context"
	"math"
	"regexp"
	"strings"
	"sync"

	"github.com/qiangli/nadir/types"
)

// Thresholds bucket a score into (simple, mid, complex). Defaults are
// 0.35 and 0.65 to match the upstream NADIRCLAW_TIER_THRESHOLDS
// default; nadir exposes them as NADIR_TIER_THRESHOLDS.
type Thresholds struct {
	Simple  float64
	Complex float64
	HasMid  bool
}

func DefaultThresholds() Thresholds {
	return Thresholds{Simple: 0.35, Complex: 0.65, HasMid: false}
}

// Heuristic is a Classifier that scores prompts without an embedding
// model. Long prompts, code-fence-heavy prompts, and prompts containing
// "complex" keywords score high; short conversational prompts score
// low. Confidence is the distance of the score from 0.5.
//
// This is a stand-in for the ONNX-backed classifier and will be
// replaced once the export-onnx tooling produces the model + centroids.
// The Classify signature, score range, and tier thresholds match the
// final implementation so callers don't need to change.
type Heuristic struct {
	thresholds Thresholds
	once       sync.Once
	complexRE  *regexp.Regexp
}

func NewHeuristic(t Thresholds) *Heuristic {
	return &Heuristic{thresholds: t}
}

func (h *Heuristic) init() {
	h.once.Do(func() {
		h.complexRE = regexp.MustCompile(`(?i)\b(refactor|implement|design|architect|debug|optimi[sz]e|analy[sz]e|prove|derive|migrate|threading|concurren|race|deadlock|recursion|reasoning|why|explain in detail|step by step)\b`)
	})
}

func (h *Heuristic) Warmup(_ context.Context) error { return nil }

func (h *Heuristic) Classify(_ context.Context, prompt string) (types.Tier, float64, float64, error) {
	h.init()

	// Three signals, each in [0, 1], averaged.
	length := lengthSignal(prompt)
	codey := codeSignal(prompt)
	keyword := keywordSignal(prompt, h.complexRE)

	score := clamp01((length + codey + keyword) / 3.0)
	confidence := math.Abs(score - 0.5) * 2 // [0,1]

	tier := BucketScore(score, h.thresholds)
	return tier, score, confidence, nil
}

// BucketScore maps a [0,1] score to a tier using the configured
// thresholds. It is exported so router tests can construct decisions
// without going through a classifier.
func BucketScore(score float64, t Thresholds) types.Tier {
	if score <= t.Simple {
		return types.TierSimple
	}
	if score >= t.Complex {
		return types.TierComplex
	}
	if t.HasMid {
		return types.TierMid
	}
	return types.TierSimple
}

func lengthSignal(s string) float64 {
	n := len(s)
	switch {
	case n < 64:
		return 0.10
	case n < 256:
		return 0.30
	case n < 1024:
		return 0.60
	case n < 4096:
		return 0.80
	default:
		return 0.95
	}
}

func codeSignal(s string) float64 {
	if strings.Contains(s, "```") {
		return 0.85
	}
	if strings.Count(s, "{") >= 3 || strings.Count(s, "(") >= 6 {
		return 0.65
	}
	return 0.20
}

func keywordSignal(s string, re *regexp.Regexp) float64 {
	matches := re.FindAllString(s, -1)
	switch {
	case len(matches) >= 2:
		return 0.90
	case len(matches) == 1:
		return 0.70
	default:
		return 0.20
	}
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

var _ types.Classifier = (*Heuristic)(nil)
