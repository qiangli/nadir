// Package compress summarises long conversations: when the message
// count or total byte size crosses a threshold, the oldest non-system
// turns are folded into a synthetic "[earlier conversation summary]"
// message so the live context window doesn't blow up.
//
// The summary is a deterministic concatenation of the first N
// characters of each folded turn — not LLM-generated. That keeps the
// proxy free of recursive LLM calls and predictable in cost. A future
// pass can plug in a "summarise via cheap model" hook behind the same
// interface.
package compress

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/qiangli/nadir/internal/types"
)

type Config struct {
	MaxMessages int // fold when message count exceeds this
	MaxBytes    int // fold when total content bytes exceed this
	KeepRecent  int // always keep this many most-recent non-system turns
}

func DefaultConfig() Config {
	return Config{MaxMessages: 40, MaxBytes: 32 * 1024, KeepRecent: 8}
}

type Result struct {
	Messages []types.Message
	Folded   int
}

func Apply(msgs []types.Message, cfg Config) Result {
	if cfg.MaxMessages == 0 {
		cfg = DefaultConfig()
	}
	if !needsCompress(msgs, cfg) {
		return Result{Messages: msgs}
	}

	// Stable system messages on top.
	systems := []types.Message{}
	conv := []types.Message{}
	for _, m := range msgs {
		if m.Role == "system" {
			systems = append(systems, m)
		} else {
			conv = append(conv, m)
		}
	}
	keep := min(cfg.KeepRecent, len(conv))
	foldIdx := len(conv) - keep
	if foldIdx <= 0 {
		return Result{Messages: msgs}
	}
	folded := conv[:foldIdx]
	recent := conv[foldIdx:]
	summary := summarize(folded)

	out := append([]types.Message{}, systems...)
	out = append(out, types.Message{
		Role:    "system",
		Content: encode(summary),
	})
	out = append(out, recent...)
	return Result{Messages: out, Folded: len(folded)}
}

func needsCompress(msgs []types.Message, cfg Config) bool {
	if cfg.MaxMessages > 0 && len(msgs) > cfg.MaxMessages {
		return true
	}
	if cfg.MaxBytes > 0 {
		n := 0
		for _, m := range msgs {
			n += len(m.Content)
		}
		if n > cfg.MaxBytes {
			return true
		}
	}
	return false
}

func summarize(msgs []types.Message) string {
	var b strings.Builder
	b.WriteString("[earlier conversation summary]\n")
	for _, m := range msgs {
		text := decode(m.Content)
		if len(text) > 200 {
			text = text[:200] + "…"
		}
		fmt.Fprintf(&b, "- %s: %s\n", m.Role, text)
	}
	return b.String()
}

func decode(raw json.RawMessage) string {
	var s string
	_ = json.Unmarshal(raw, &s)
	return s
}

func encode(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}
