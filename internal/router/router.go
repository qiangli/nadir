// Package router resolves a ChatRequest into a RouteDecision: which
// model to call, the fallback chain, the classifier score, and any
// modifiers applied. The pipeline mirrors
// priorart/NadirClaw/nadirclaw/routing.py:
//
//   profile → alias → classify → modifiers (agentic / reasoning /
//   vision) → session-cache upgrade → provider-health reorder.
package router

import (
	"context"
	"encoding/json"
	"regexp"
	"slices"
	"strings"

	"github.com/qiangli/nadir/internal/classifier"
	"github.com/qiangli/nadir/internal/config"
	"github.com/qiangli/nadir/internal/types"
)

type Router struct {
	cfg        *config.Config
	classifier types.Classifier
	session    types.SessionCache
	health     types.ProviderHealth

	thresholds classifier.Thresholds
}

func New(cfg *config.Config, c types.Classifier, sess types.SessionCache, h types.ProviderHealth) *Router {
	t := classifier.Thresholds{
		Simple:  cfg.TierThresholds[0],
		Complex: cfg.TierThresholds[1],
		HasMid:  cfg.MidModel != "",
	}
	return &Router{cfg: cfg, classifier: c, session: sess, health: h, thresholds: t}
}

func (r *Router) Route(ctx context.Context, req *types.ChatRequest, _ *types.User) (*types.RouteDecision, error) {
	meta := extractMeta(req)
	modifiers := []string{}

	// 1. Profile / alias / explicit-model handling.
	requested := strings.ToLower(strings.TrimSpace(req.Model))
	var primary string
	var tier types.Tier
	var score, confidence float64

	switch requested {
	case "", "auto", "smart":
		// Classify the prompt and pick a tier model.
		t, s, conf, err := r.classifier.Classify(ctx, meta.UserText)
		if err != nil {
			return nil, err
		}
		tier, score, confidence = t, s, conf
		primary = r.modelForTier(tier)
		modifiers = append(modifiers, "classified")
	default:
		// Treat as an explicit model name; aliases handled later.
		primary = r.resolveAlias(req.Model)
		tier = r.tierForModel(primary)
	}

	// 2. Modifiers — agentic/reasoning/vision can bump the tier.
	if meta.HasTools || meta.HasAgentRole {
		if up := r.upgradeTier(primary, tier, types.TierComplex); up != primary {
			primary = up
			tier = types.TierComplex
			modifiers = append(modifiers, "agentic")
		}
	}
	if meta.WantsReasoning && r.cfg.ReasoningModel != "" {
		primary = r.cfg.ReasoningModel
		tier = types.TierReasoning
		modifiers = append(modifiers, "reasoning")
	}
	if meta.HasImages {
		if !modelHasVision(primary) {
			if v := r.pickVisionModel(); v != "" {
				primary = v
				modifiers = append(modifiers, "vision_swap")
			}
		}
	}

	// 3. Session pinning — upgrade to the recorded model if higher.
	if r.session != nil {
		final, finalTier, status := r.session.UpgradeIfHigher(req.Messages, primary, tier)
		if status == "pinned" || status == "upgraded" {
			modifiers = append(modifiers, "session_"+status)
			primary = final
			tier = finalTier
		}
	}

	// 4. Fallback chain: configured chain plus tier ladder, dedup,
	//    drop the primary, then reorder by provider health.
	chain := r.buildChain(primary, tier)

	provider := r.cfg.ProviderForModel[primary]
	if provider == "" {
		provider = "openai"
	}

	return &types.RouteDecision{
		Model:         primary,
		Tier:          tier,
		Provider:      provider,
		Score:         score,
		Confidence:    confidence,
		Modifiers:     modifiers,
		FallbackChain: chain,
	}, nil
}

func (r *Router) modelForTier(t types.Tier) string {
	switch t {
	case types.TierSimple:
		return r.cfg.SimpleModel
	case types.TierMid:
		if r.cfg.MidModel != "" {
			return r.cfg.MidModel
		}
		return r.cfg.SimpleModel
	case types.TierComplex:
		return r.cfg.ComplexModel
	case types.TierReasoning:
		if r.cfg.ReasoningModel != "" {
			return r.cfg.ReasoningModel
		}
		return r.cfg.ComplexModel
	default:
		return r.cfg.SimpleModel
	}
}

func (r *Router) tierForModel(model string) types.Tier {
	switch model {
	case r.cfg.SimpleModel:
		return types.TierSimple
	case r.cfg.MidModel:
		return types.TierMid
	case r.cfg.ComplexModel:
		return types.TierComplex
	case r.cfg.ReasoningModel:
		return types.TierReasoning
	}
	return types.TierMid
}

// modelAliases mirrors priorart/NadirClaw/nadirclaw/routing.py
// (MODEL_ALIASES). Keep in sync via the parity test.
var modelAliases = map[string]string{
	"sonnet":            "claude-sonnet-4-5-20250929",
	"opus":              "claude-opus-4-6-20250918",
	"haiku":             "claude-haiku-4-5-20251001",
	"claude":            "claude-sonnet-4-5-20250929",
	"gpt4":              "gpt-4.1",
	"gpt4o":             "gpt-4o",
	"gpt4-mini":         "gpt-4.1-mini",
	"gpt5":              "gpt-5.2",
	"gpt5-mini":         "gpt-5-mini",
	"o3":                "o3",
	"o3-mini":           "o3-mini",
	"o4-mini":           "o4-mini",
	"flash":             "gemini-2.5-flash",
	"gemini-flash":      "gemini-2.5-flash",
	"gemini-pro":        "gemini-2.5-pro",
	"deepseek":          "deepseek/deepseek-chat",
	"deepseek-v4":       "deepseek/deepseek-v4-flash",
	"deepseek-v4-flash": "deepseek/deepseek-v4-flash",
	"deepseek-v4-pro":   "deepseek/deepseek-v4-pro",
	"deepseek-r1":       "deepseek/deepseek-reasoner",
	"llama":             "ollama/llama3.1:8b",
}

func (r *Router) resolveAlias(model string) string {
	if resolved, ok := modelAliases[strings.ToLower(strings.TrimSpace(model))]; ok {
		return resolved
	}
	return model
}

func (r *Router) upgradeTier(current string, currentTier, want types.Tier) string {
	if tierRank(currentTier) >= tierRank(want) {
		return current
	}
	return r.modelForTier(want)
}

func (r *Router) pickVisionModel() string {
	// Best-effort: prefer the complex model if it looks vision-capable.
	if modelHasVision(r.cfg.ComplexModel) {
		return r.cfg.ComplexModel
	}
	return ""
}

func (r *Router) buildChain(primary string, tier types.Tier) []string {
	seen := map[string]bool{primary: true}
	out := []string{}
	add := func(m string) {
		if m == "" || seen[m] {
			return
		}
		seen[m] = true
		out = append(out, m)
	}

	// Configured fallback chain first.
	for _, m := range r.cfg.FallbackChain {
		add(m)
	}
	// Tier ladder: progressively cheaper alternatives.
	switch tier {
	case types.TierReasoning, types.TierComplex:
		add(r.cfg.MidModel)
		add(r.cfg.SimpleModel)
	case types.TierMid:
		add(r.cfg.SimpleModel)
	}

	if r.health != nil {
		out = r.health.Order(out)
	}
	return out
}

func tierRank(t types.Tier) int {
	switch t {
	case types.TierFree:
		return 0
	case types.TierSimple:
		return 1
	case types.TierMid:
		return 2
	case types.TierComplex:
		return 3
	case types.TierReasoning:
		return 4
	}
	return 0
}

// modelHasVision is a best-effort lookup keyed off model name. A real
// table (modelmeta package) lands in Phase 2; for now we keep it
// simple so the router compiles and tests pass.
func modelHasVision(model string) bool {
	m := strings.ToLower(model)
	return strings.Contains(m, "4o") ||
		strings.Contains(m, "claude") ||
		strings.Contains(m, "gemini") ||
		strings.Contains(m, "vision")
}

// meta is the local extracted-request bag — duplicated from types.RequestMeta
// with router-specific fields so we don't widen the public type.
type meta struct {
	UserText      string
	SystemText    string
	HasImages     bool
	HasTools      bool
	HasAgentRole  bool
	WantsReasoning bool
}

func extractMeta(req *types.ChatRequest) meta {
	m := meta{HasTools: len(req.Tools) > 0}
	for _, msg := range req.Messages {
		text := contentText(msg.Content)
		switch msg.Role {
		case "system":
			m.SystemText += text + "\n"
			if containsAgentRoleHint(text) {
				m.HasAgentRole = true
			}
		case "user":
			m.UserText = text // keep last user message
			if containsImagePart(msg.Content) {
				m.HasImages = true
			}
			if containsReasoningHint(text) {
				m.WantsReasoning = true
			}
		case "tool":
			m.HasTools = true
		}
		if slices.ContainsFunc(msg.ToolCalls, func(tc types.ToolCall) bool { return tc.ID != "" }) {
			m.HasTools = true
		}
	}
	return m
}

// contentText extracts plain text from a Message.Content payload.
// Content can be a JSON string or an array of {type,text/image_url} parts.
func contentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" && p.Text != "" {
				if b.Len() > 0 {
					b.WriteString("\n")
				}
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return string(raw)
}

func containsImagePart(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var parts []map[string]any
	if err := json.Unmarshal(raw, &parts); err != nil {
		return false
	}
	for _, p := range parts {
		t, _ := p["type"].(string)
		if t == "image" || t == "image_url" {
			return true
		}
	}
	return false
}

func containsAgentRoleHint(systemText string) bool {
	s := strings.ToLower(systemText)
	return strings.Contains(s, "you are an agent") ||
		strings.Contains(s, "agentic") ||
		strings.Contains(s, "you can call tools") ||
		strings.Contains(s, "autonomous")
}

// reasoningMarkers mirrors NadirClaw's _REASONING_MARKERS_EN regex
// (priorart/NadirClaw/nadirclaw/routing.py). Upstream requires the
// combined prompt + system_message to match at least one marker for
// is_reasoning=true. We follow the same contract.
var reasoningMarkers = regexp.MustCompile(`(?i)\b(` +
	`step[- ]by[- ]step` +
	`|think (?:through|carefully|deeply|about)` +
	`|chain[- ]of[- ]thought` +
	`|let'?s? reason` +
	`|reason(?:ing)? (?:about|through)` +
	`|prove (?:that|this|the)` +
	`|formal (?:proof|verification)` +
	`|mathematical(?:ly)? (?:prove|show|derive)` +
	`|derive (?:the|a|an)` +
	`|analyze the (?:tradeoffs?|trade-offs?|implications?|consequences?)` +
	`|compare and contrast` +
	`|what are the (?:pros? and cons?|advantages? and disadvantages?)` +
	`|evaluate (?:the|whether|if)` +
	`|critically (?:analyze|assess|examine)` +
	`|explain (?:why|how|the reasoning)` +
	`|work through` +
	`|break (?:this|it) down` +
	`|logical(?:ly)? (?:deduce|infer|conclude)` +
	`|analyze why (?:this|the|it)` +
	`|diagnose the (?:root )?cause` +
	`|weigh (?:the )?(?:pros|cons|options|alternatives)` +
	`|architectural (?:decision|choice)` +
	`|design (?:a )?(?:system|architecture)` +
	`)\b`)

// containsReasoningHint reports whether the combined prompt + system
// text crosses NadirClaw's 2-markers threshold (priorart/.../routing.py
// detect_reasoning). Single-marker prompts like "Explain why X" are
// intentionally NOT classified as reasoning — the upstream tests use
// this exact threshold so we mirror it.
// codeReviewMarkers mirrors NadirClaw's _REVIEW_MARKERS regex.
var codeReviewMarkers = regexp.MustCompile(`(?i)(code\s*review|review\s*(?:the\s+)?(?:code|changes|pr|diff)|pull\s*request\s*review|security\s*(?:audit|review)|static\s*analysis|lint\s*check)`)

// detectCodeReview reports whether the prompt asks for code-review-shaped
// work. The Python upstream's detect_code_review returns a confidence
// score with a 0.80 threshold; we collapse it to a bool match.
func detectCodeReview(prompt, systemMessage string) bool {
	text := prompt
	if systemMessage != "" {
		text = systemMessage + "\n" + prompt
	}
	return codeReviewMarkers.MatchString(text)
}

func containsReasoningHint(text string) bool {
	if text == "" {
		return false
	}
	matches := reasoningMarkers.FindAllString(text, -1)
	if len(matches) < 2 {
		return false
	}
	// Dedup: NadirClaw's `set()` collapses identical markers found
	// multiple times before applying the >= 2 threshold.
	seen := make(map[string]bool, len(matches))
	for _, m := range matches {
		seen[strings.ToLower(m)] = true
	}
	return len(seen) >= 2
}

var _ types.Router = (*Router)(nil)
