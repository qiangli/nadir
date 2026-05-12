package router

// Config is the library-facing configuration for Router. It contains
// only the fields the router actually consumes; the binary's larger
// env-loading config in internal/config produces one of these via
// (*config.Config).RouterConfig().
//
// Library users construct this directly:
//
//	cfg := &router.Config{
//	    SimpleModel:    "claude-haiku-4-5-20251001",
//	    ComplexModel:   "claude-opus-4-6-20250918",
//	    TierThresholds: [2]float64{0.35, 0.65},
//	}
type Config struct {
	// Tier-model assignments. SimpleModel is required; the others
	// can be empty to fall back to SimpleModel.
	SimpleModel    string
	MidModel       string
	ComplexModel   string
	ReasoningModel string

	// FallbackChain is the ordered list of models tried after the
	// primary in the dispatch loop. The router doesn't actually
	// dispatch — it just composes the chain — so the proxy layer is
	// what actually walks this list.
	FallbackChain []string

	// TierThresholds bucket the classifier score:
	//   score ≤ TierThresholds[0]                       → simple
	//   TierThresholds[0] < score < TierThresholds[1]   → mid (if configured) else simple
	//   score ≥ TierThresholds[1]                       → complex
	TierThresholds [2]float64

	// ProviderForModel maps a model name to the provider key used
	// by the proxy's LLMClient lookup. The router itself only
	// reads this to populate RouteDecision.Provider.
	ProviderForModel map[string]string
}
