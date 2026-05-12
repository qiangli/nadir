package main

import (
	"log/slog"
	"time"

	"github.com/qiangli/nadir/classifier"
	"github.com/qiangli/nadir/internal/config"
	"github.com/qiangli/nadir/provider/openai"
	"github.com/qiangli/nadir/types"
)

// activeClassifier carries the assembled classifier plus a short
// label ("heuristic" / "onnx" / "cascade(onnx+llama3.2:3b)") that
// /health exposes for monitoring. The label lets dashboards alert
// when a deploy silently flips to a degraded mode.
type activeClassifier struct {
	Classifier types.Classifier
	Label      string
}

// buildClassifier assembles the runtime classifier. The two build
// tags express production intent:
//
//   - Default build (no `-tags onnx`): heuristic only. Suitable for
//     dev, tests, CI smoke checks. No ML deps, no CGO, no setup.
//   - `-tags onnx` build: ONNX MiniLM is REQUIRED. If assets are
//     missing or libonnxruntime can't be loaded, startup fails fast.
//     There is no silent fallback to heuristic in production builds.
//
// The build tag is the strict-mode switch — we don't add an env-var
// flag for "is this production" because the build pipeline already
// knows. If you compiled `-tags onnx`, you opted into the runtime
// requirement.
//
// Returned error halts `nadir serve` startup with a clear message
// rather than letting the proxy boot in a quietly-degraded state.
func buildClassifier(logger *slog.Logger, cfg *config.Config, thresh classifier.Thresholds) (*activeClassifier, error) {
	primary, primaryLabel, err := buildPrimary(logger, thresh)
	if err != nil {
		return nil, err
	}

	// Cascade wrapper: ask a small LLM for a second opinion when
	// primary confidence is below threshold. Disabled when threshold
	// is 0 or no LLM model is configured.
	if cfg.CascadeThreshold > 0 && cfg.CascadeLLMModel != "" {
		baseURL := cfg.CascadeLLMBaseURL
		if baseURL == "" {
			baseURL = cfg.OllamaBaseURL
		}
		llmClient := openai.New("cascade-llm", baseURL, cfg.CascadeLLMAPIKey)
		secondary := classifier.NewLLM(llmClient, cfg.CascadeLLMModel, thresh)
		c := classifier.NewCascading(primary, secondary, cfg.CascadeThreshold)
		c.Logger = logger
		c.SecondaryTimeout = time.Duration(cfg.CascadeLLMTimeoutSec) * time.Second
		logger.Info("classifier: cascading",
			slog.String("primary", primaryLabel),
			slog.Float64("threshold", cfg.CascadeThreshold),
			slog.String("llm_model", cfg.CascadeLLMModel),
			slog.String("llm_base_url", baseURL),
		)
		return &activeClassifier{
			Classifier: c,
			Label:      "cascade(" + primaryLabel + "+" + cfg.CascadeLLMModel + ")",
		}, nil
	}
	return &activeClassifier{Classifier: primary, Label: primaryLabel}, nil
}
