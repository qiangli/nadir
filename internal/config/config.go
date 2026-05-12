// Package config loads nadir runtime settings from env vars (and,
// later, ~/.nadir/.env). Env-var contract is documented in README and
// maps semantically to the upstream NADIRCLAW_* keys (one-time
// migration is a sed-style rename).
package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/qiangli/nadir/router"
)

// RouterConfig projects the binary's superset config down to the
// library-facing router.Config that the Router actually consumes.
func (c *Config) RouterConfig() *router.Config {
	return &router.Config{
		SimpleModel:      c.SimpleModel,
		MidModel:         c.MidModel,
		ComplexModel:     c.ComplexModel,
		ReasoningModel:   c.ReasoningModel,
		FallbackChain:    c.FallbackChain,
		TierThresholds:   c.TierThresholds,
		ProviderForModel: c.ProviderForModel,
	}
}

type Config struct {
	// Server
	Addr      string
	AuthToken string
	MaxBodyMB int

	// Routing
	SimpleModel    string
	MidModel       string
	ComplexModel   string
	ReasoningModel string
	FallbackChain  []string
	TierThresholds [2]float64 // simple ceiling, complex floor

	// Rate limiting
	UserRateWindow time.Duration
	UserRateLimit  int

	// Cache
	CacheMaxSize int
	CacheTTL     time.Duration

	// Providers
	OpenAIBaseURL   string
	OpenAIAPIKey    string
	AnthropicAPIKey string
	GoogleAPIKey    string
	OllamaBaseURL   string

	// Cascading classifier (defense-in-depth: ask a small LLM for a
	// second opinion when the primary embedding/heuristic is uncertain).
	// Disabled when threshold is 0 or model is empty.
	CascadeThreshold     float64
	CascadeLLMModel      string
	CascadeLLMBaseURL    string
	CascadeLLMAPIKey     string
	CascadeLLMTimeoutSec int

	// Provider routing: model name → provider name (one of "openai",
	// "anthropic", "gemini", "ollama"). Unknown models default to
	// "openai" so any OpenAI-compatible endpoint works out of the box.
	ProviderForModel map[string]string
}

// Load reads NADIR_* env vars and returns a populated Config.
// Missing keys fall back to defaults; the only hard requirement at
// startup is that at least one provider key (or Ollama URL) is set so
// the proxy has somewhere to dispatch — that check lives in
// cmd/nadir/serve.go because tests should be able to construct
// stand-alone configs.
func Load() *Config {
	c := &Config{
		Addr:            getEnv("NADIR_ADDR", ":8856"),
		AuthToken:       os.Getenv("NADIR_AUTH_TOKEN"),
		MaxBodyMB:       getEnvInt("NADIR_MAX_BODY_MB", 20),
		SimpleModel:     getEnv("NADIR_SIMPLE_MODEL", "gpt-4o-mini"),
		MidModel:        os.Getenv("NADIR_MID_MODEL"),
		ComplexModel:    getEnv("NADIR_COMPLEX_MODEL", "gpt-4o"),
		ReasoningModel:  os.Getenv("NADIR_REASONING_MODEL"),
		FallbackChain:   splitComma(os.Getenv("NADIR_FALLBACK_CHAIN")),
		TierThresholds:  getEnvThresholds("NADIR_TIER_THRESHOLDS", [2]float64{0.35, 0.65}),
		UserRateWindow:  getEnvDuration("NADIR_RATE_WINDOW", time.Minute),
		UserRateLimit:   getEnvInt("NADIR_RATE_LIMIT", 120),
		CacheMaxSize:    getEnvInt("NADIR_CACHE_MAX_SIZE", 256),
		CacheTTL:        getEnvDuration("NADIR_CACHE_TTL", 30*time.Minute),
		OpenAIBaseURL:   os.Getenv("NADIR_OPENAI_BASE_URL"),
		OpenAIAPIKey:    os.Getenv("NADIR_OPENAI_API_KEY"),
		AnthropicAPIKey: os.Getenv("NADIR_ANTHROPIC_API_KEY"),
		GoogleAPIKey:    os.Getenv("NADIR_GOOGLE_API_KEY"),
		OllamaBaseURL:   getEnv("NADIR_OLLAMA_BASE_URL", "http://localhost:11434/v1"),

		// Cascade defaults: OFF unless CASCADE_THRESHOLD > 0. When
		// enabled, the LLM model defaults to a small Ollama model so
		// the second opinion is cheap.
		CascadeThreshold:     getEnvFloat("NADIR_CASCADE_THRESHOLD", 0),
		CascadeLLMModel:      os.Getenv("NADIR_CASCADE_LLM_MODEL"),
		CascadeLLMBaseURL:    os.Getenv("NADIR_CASCADE_LLM_BASE_URL"),
		CascadeLLMAPIKey:     os.Getenv("NADIR_CASCADE_LLM_API_KEY"),
		CascadeLLMTimeoutSec: getEnvInt("NADIR_CASCADE_LLM_TIMEOUT_SEC", 2),
	}
	c.ProviderForModel = inferProviderMap(c)
	return c
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getEnvInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func getEnvFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func getEnvDuration(key string, def time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}

func getEnvThresholds(key string, def [2]float64) [2]float64 {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	parts := strings.Split(v, ",")
	if len(parts) != 2 {
		fmt.Fprintf(os.Stderr, "%s: want two comma-separated floats, got %q; using defaults\n", key, v)
		return def
	}
	out := def
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s[%d]=%q: %v; using default\n", key, i, p, err)
			return def
		}
		out[i] = f
	}
	return out
}

func splitComma(v string) []string {
	if v == "" {
		return nil
	}
	out := strings.Split(v, ",")
	for i, s := range out {
		out[i] = strings.TrimSpace(s)
	}
	return out
}

// inferProviderMap is a best-effort guess: model names containing
// "claude" → anthropic, "gemini" → gemini, "ollama/" prefix → ollama,
// everything else → openai. Explicit configuration overrides this in
// NADIR_PROVIDER_MAP (key=value,key=value) for ambiguous cases.
func inferProviderMap(c *Config) map[string]string {
	out := make(map[string]string)
	consider := []string{c.SimpleModel, c.MidModel, c.ComplexModel, c.ReasoningModel}
	consider = append(consider, c.FallbackChain...)
	for _, m := range consider {
		if m == "" {
			continue
		}
		out[m] = providerFor(m)
	}
	for _, entry := range splitComma(os.Getenv("NADIR_PROVIDER_MAP")) {
		k, v, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		out[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	return out
}

func providerFor(model string) string {
	m := strings.ToLower(model)
	switch {
	case strings.HasPrefix(m, "ollama/"):
		return "ollama"
	case strings.Contains(m, "claude"):
		return "anthropic"
	case strings.Contains(m, "gemini"):
		return "gemini"
	default:
		return "openai"
	}
}
