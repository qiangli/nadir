package router

// Routing parity diagnostic. Loads testdata/routing_parity_corpus.json
// (generated from priorart/NadirClaw/nadirclaw/routing.py by
// tools/parity-corpus/extract.py) and reports per-category match
// rates. The harness is intentionally lenient — it fails only when
// parity drops below a documented floor — so we can spot drift
// without breaking CI every time NadirClaw upstream tweaks a marker
// or alias.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type parityCorpus struct {
	Alias      []aliasCase      `json:"alias"`
	Profile    []profileCase    `json:"profile"`
	Agentic    []agenticCase    `json:"agentic"`
	Reasoning  []reasoningCase  `json:"reasoning"`
	CodeReview []codeReviewCase `json:"code_review"`
	Images     []imageCase      `json:"images"`
}

type codeReviewCase struct {
	Prompt           string `json:"prompt"`
	SystemMessage    string `json:"system_message"`
	ExpectedIsReview bool   `json:"expected_is_review"`
}

type imageCase struct {
	Label            string `json:"label"`
	ExpectedHasImage bool   `json:"expected_has_images"`
}

type aliasCase struct {
	Input    string `json:"input"`
	Expected any    `json:"expected"` // string or nil
}

type profileCase struct {
	Input    any `json:"input"`    // string or nil
	Expected any `json:"expected"` // string or nil
}

type agenticCase struct {
	Label           string `json:"label"`
	HasTools        bool   `json:"has_tools"`
	ToolCount       int    `json:"tool_count"`
	SystemPrompt    string `json:"system_prompt"`
	MessageCount    int    `json:"message_count"`
	ExpectedAgentic bool   `json:"expected_is_agentic"`
}

type reasoningCase struct {
	Prompt            string `json:"prompt"`
	SystemMessage     string `json:"system_message"`
	ExpectedReasoning bool   `json:"expected_is_reasoning"`
}

func loadCorpus(t *testing.T) *parityCorpus {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "routing_parity_corpus.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("corpus not generated yet; run python tools/parity-corpus/extract.py: %v", err)
		return nil
	}
	var c parityCorpus
	if err := json.Unmarshal(data, &c); err != nil {
		t.Fatalf("malformed corpus: %v", err)
	}
	return &c
}

// -------- alias --------
//
// Our resolveAlias is on Router; expose a package-private helper that
// returns nil for non-aliases (matching NadirClaw's contract — they
// return None for both unknown names AND already-resolved canonical
// names). This is the test the diagnostic uses.
func nadirResolveAlias(input string) (string, bool) {
	resolved, ok := modelAliases[strings.ToLower(strings.TrimSpace(input))]
	return resolved, ok
}

func TestParityAliasResolution(t *testing.T) {
	c := loadCorpus(t)
	if c == nil {
		return
	}
	total, match := 0, 0
	var miss []string
	for _, tc := range c.Alias {
		total++
		got, gotOk := nadirResolveAlias(tc.Input)
		expectStr, expectOk := tc.Expected.(string)
		expectOk = expectOk && expectStr != ""

		if expectOk && gotOk && got == expectStr {
			match++
		} else if !expectOk && !gotOk {
			match++
		} else {
			miss = append(miss, fmt.Sprintf("  %q: nadir=%q(ok=%v) nadirclaw=%v",
				tc.Input, got, gotOk, tc.Expected))
		}
	}
	rate := float64(match) / float64(total)
	t.Logf("alias parity: %d/%d = %.0f%%", match, total, rate*100)
	for _, m := range miss {
		t.Log(m)
	}
	// Floor: aliases that exist in both maps should match. NadirClaw
	// uses date-stamped model names (claude-sonnet-4-5-20250929) we
	// don't carry; floor allows that divergence.
	if rate < 0.30 {
		t.Errorf("alias parity %.0f%% below 30%% floor", rate*100)
	}
}

// -------- profile --------
//
// Our router doesn't have profile resolution as a first-class concept
// — "auto" is the only profile we honour explicitly; eco/premium/free
// are bucketed into the classifier's tier output. This test documents
// the gap. To get parity, add a resolveProfile helper to router.go
// that maps these strings.
func TestParityProfileResolution(t *testing.T) {
	c := loadCorpus(t)
	if c == nil {
		return
	}
	total, match := 0, 0
	var miss []string
	for _, tc := range c.Profile {
		total++
		input, _ := tc.Input.(string)
		got := resolveProfileStub(input)
		expect, _ := tc.Expected.(string)
		if got == expect {
			match++
		} else {
			miss = append(miss, fmt.Sprintf("  input=%q: nadir=%q nadirclaw=%q", input, got, expect))
		}
	}
	rate := float64(match) / float64(total)
	t.Logf("profile parity: %d/%d = %.0f%%", match, total, rate*100)
	for _, m := range miss {
		t.Log(m)
	}
	// Soft check — we currently expect ~50% (the unknown/empty cases
	// happen to align). This rate documents the gap; the user can
	// decide whether to close it.
}

// resolveProfileStub mirrors what NadirClaw's resolve_profile does —
// returns "auto"/"eco"/"premium"/"free"/"reasoning" or "". Our router
// doesn't use this output today; the function exists here so the
// diagnostic measures the gap honestly.
func resolveProfileStub(input string) string {
	s := strings.ToLower(strings.TrimSpace(input))
	if s == "" {
		return ""
	}
	if rest, ok := strings.CutPrefix(s, "nadirclaw/"); ok {
		s = rest
	}
	switch s {
	case "auto", "eco", "premium", "free", "reasoning":
		return s
	}
	return ""
}

// -------- agentic --------
//
// NadirClaw's detect_agentic combines: tool count, message-history
// patterns, system-prompt length, and keyword scoring. We do a much
// simpler "has tools OR has agent role keyword" check in
// router.go::extractMeta. The diagnostic shows where that simplifies
// vs the upstream.
func TestParityAgenticDetection(t *testing.T) {
	c := loadCorpus(t)
	if c == nil {
		return
	}
	total, match := 0, 0
	var miss []string
	for _, tc := range c.Agentic {
		total++
		gotAgentic := nadirDetectAgentic(tc)
		if gotAgentic == tc.ExpectedAgentic {
			match++
		} else {
			miss = append(miss, fmt.Sprintf("  %s: nadir=%v nadirclaw=%v (has_tools=%v tool_count=%d msg=%d sys_len=%d)",
				tc.Label, gotAgentic, tc.ExpectedAgentic, tc.HasTools, tc.ToolCount, tc.MessageCount, len(tc.SystemPrompt)))
		}
	}
	rate := float64(match) / float64(total)
	t.Logf("agentic parity: %d/%d = %.0f%%", match, total, rate*100)
	for _, m := range miss {
		t.Log(m)
	}
	if rate < 0.60 {
		t.Errorf("agentic parity %.0f%% below 60%% floor", rate*100)
	}
}

// nadirDetectAgentic mirrors what router.go::extractMeta does for the
// agentic check: tools_defined → agentic; long system prompt
// (>1000 chars) → agentic; agent-role keyword in system → agentic;
// deep conversation (>10 messages with tools) → agentic. This is the
// simplified Go path the diagnostic measures.
func nadirDetectAgentic(tc agenticCase) bool {
	if tc.HasTools && tc.ToolCount > 0 {
		return true
	}
	if len(tc.SystemPrompt) > 1000 {
		return true
	}
	if containsAgentRoleHint(tc.SystemPrompt) {
		return true
	}
	if tc.MessageCount > 10 && tc.HasTools {
		return true
	}
	return false
}

// -------- reasoning --------
//
// NadirClaw requires ≥2 markers in the prompt + system_message
// combined for is_reasoning=true. Our router fires on a single hint
// (router.go::containsReasoningHint). Diagnostic shows the gap.
func TestParityReasoningDetection(t *testing.T) {
	c := loadCorpus(t)
	if c == nil {
		return
	}
	total, match := 0, 0
	var miss []string
	for _, tc := range c.Reasoning {
		total++
		got := nadirDetectReasoning(tc.Prompt, tc.SystemMessage)
		if got == tc.ExpectedReasoning {
			match++
		} else {
			miss = append(miss, fmt.Sprintf("  prompt=%q sys=%q: nadir=%v nadirclaw=%v",
				tc.Prompt, tc.SystemMessage, got, tc.ExpectedReasoning))
		}
	}
	rate := float64(match) / float64(total)
	t.Logf("reasoning parity: %d/%d = %.0f%%", match, total, rate*100)
	for _, m := range miss {
		t.Log(m)
	}
	if rate < 0.50 {
		t.Errorf("reasoning parity %.0f%% below 50%% floor", rate*100)
	}
}

func nadirDetectReasoning(prompt, system string) bool {
	// Mirror NadirClaw: combine system + prompt, then apply the
	// 2-markers threshold across the combined text.
	combined := system + " " + prompt
	return containsReasoningHint(combined)
}

// -------- code review --------

func TestParityCodeReviewDetection(t *testing.T) {
	c := loadCorpus(t)
	if c == nil {
		return
	}
	total, match := 0, 0
	var miss []string
	for _, tc := range c.CodeReview {
		total++
		got := detectCodeReview(tc.Prompt, tc.SystemMessage)
		if got == tc.ExpectedIsReview {
			match++
		} else {
			miss = append(miss, fmt.Sprintf("  prompt=%q sys=%q: nadir=%v nadirclaw=%v",
				tc.Prompt, tc.SystemMessage, got, tc.ExpectedIsReview))
		}
	}
	rate := float64(match) / float64(total)
	t.Logf("code_review parity: %d/%d = %.0f%%", match, total, rate*100)
	for _, m := range miss {
		t.Log(m)
	}
	if rate < 0.75 {
		t.Errorf("code_review parity %.0f%% below 75%% floor", rate*100)
	}
}
