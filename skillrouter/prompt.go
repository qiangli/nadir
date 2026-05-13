package skillrouter

import (
	"encoding/json"
	"fmt"
	"strings"
)

// promptHeader and promptFooter wrap the skill catalog. The shape
// mirrors classifier.llmClassifyPrompt — a short instruction, a
// constrained reply format, then "Request:\n%s\n<label>:" so even
// small instruction-tuned models stay on-task.
const (
	promptHeader = `You are a skill router. Pick the skill that best handles the user's request from the list below.

Reply with EXACTLY the skill name (e.g., %s) and nothing else. If no skill applies, reply with the single word: none.

Skills:
`
	promptFooter = `
Request:
%s

Skill:`
)

// renderPrompt builds the user-message body. The first skill's Name
// goes into the header as a worked example so the model copies its
// format (slashes, casing, etc.).
func renderPrompt(skills []Skill, prompt string) string {
	var b strings.Builder
	example := ""
	if len(skills) > 0 {
		example = skills[0].Name
	}
	fmt.Fprintf(&b, promptHeader, example)
	for _, s := range skills {
		desc := s.Description
		if desc == "" {
			fmt.Fprintf(&b, "- %s\n", s.Name)
		} else {
			fmt.Fprintf(&b, "- %s — %s\n", s.Name, desc)
		}
	}
	fmt.Fprintf(&b, promptFooter, truncatePrompt(prompt, 4000))
	return b.String()
}

// parseSkill maps the LLM's raw reply to a catalog entry. Match
// precedence — most specific first, so a model that echoes the
// skill name plus a stray period still wins exact-equal:
//
//  1. exact match against any Name             (confidence 1.0)
//  2. case-insensitive match against any Name  (confidence 0.9)
//  3. catalog Name is a substring of the reply (confidence 0.7)
//
// "none" (case-insensitive) and any unparseable reply produce
// FellThrough=true with Confidence=0. Empty replies fall through.
//
// Substring matching iterates the catalog in order, so if two skill
// names overlap (e.g., "/review" and "/security-review") and the LLM
// replies with the longer one, we still want the longer one to win.
// We resolve that by scoring every catalog hit and picking the
// longest match.
func parseSkill(raw string, skills []Skill) *Decision {
	cleaned := cleanReply(raw)
	if cleaned == "" {
		return &Decision{FellThrough: true}
	}
	if strings.EqualFold(cleaned, "none") {
		return &Decision{FellThrough: true}
	}

	for _, s := range skills {
		if cleaned == s.Name {
			return &Decision{Skill: s.Name, Confidence: 1.0}
		}
	}
	for _, s := range skills {
		if strings.EqualFold(cleaned, s.Name) {
			return &Decision{Skill: s.Name, Confidence: 0.9}
		}
	}

	// Substring fallback: pick the longest skill name that appears in
	// the reply. Longest-match resolves the "/review" vs
	// "/security-review" overlap correctly without needing a separate
	// disambiguation step.
	lower := strings.ToLower(cleaned)
	var best Skill
	for _, s := range skills {
		if s.Name == "" {
			continue
		}
		if strings.Contains(lower, strings.ToLower(s.Name)) && len(s.Name) > len(best.Name) {
			best = s
		}
	}
	if best.Name != "" {
		return &Decision{Skill: best.Name, Confidence: 0.7}
	}
	return &Decision{FellThrough: true}
}

// cleanReply strips the formatting cruft small models tend to emit
// around short answers: leading "Skill:" prefixes, code fences,
// quotes, trailing punctuation, surrounding whitespace.
func cleanReply(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	// Strip a leading "Skill:" / "skill:" prefix (the prompt ends with
	// "Skill:" so small models sometimes echo the label).
	if idx := strings.IndexByte(s, ':'); idx >= 0 && idx < 16 {
		prefix := strings.ToLower(s[:idx])
		if prefix == "skill" || prefix == "answer" {
			s = strings.TrimSpace(s[idx+1:])
		}
	}
	s = strings.Trim(s, "`*\"' ")
	s = strings.TrimRight(s, ".,;:!?")
	s = strings.TrimSpace(s)
	// Take the first line only — guards against models that justify
	// their pick on a second line.
	if idx := strings.IndexAny(s, "\n\r"); idx >= 0 {
		s = strings.TrimSpace(s[:idx])
		s = strings.Trim(s, "`*\"' ")
		s = strings.TrimRight(s, ".,;:!?")
	}
	return s
}

// extractText pulls the assistant content out of a JSON-encoded
// Message.Content. The shape matches classifier.parseScore's input:
// the upstream may return a plain string or a content-parts array,
// but for the small completions we ask for here a string is universal.
func extractText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	// Fallback: best-effort string of the raw bytes.
	return string(raw)
}

func jsonString(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

func truncatePrompt(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
