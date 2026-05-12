// Package optimize compacts a message list before dispatch: strips
// redundant whitespace, drops null-effect role messages, deduplicates
// repeated tool-output prefixes. In "aggressive" mode it also
// truncates the middle of long tool outputs.
package optimize

import (
	"encoding/json"
	"regexp"
	"strings"

	"github.com/qiangli/nadir/internal/types"
)

type Mode string

const (
	Off        Mode = "off"
	Safe       Mode = "safe"
	Aggressive Mode = "aggressive"
)

type Result struct {
	Messages    []types.Message
	OriginalLen int
	NewLen      int
	BytesSaved  int
	Applied     []string
}

func Apply(msgs []types.Message, mode Mode) Result {
	out := Result{Messages: msgs, OriginalLen: lenOf(msgs)}
	if mode == Off || mode == "" {
		out.NewLen = out.OriginalLen
		return out
	}
	cloned := cloneMessages(msgs)

	// Safe transforms.
	cloned = collapseWhitespace(cloned, &out)
	cloned = dedupRepeatedToolOutputs(cloned, &out)

	if mode == Aggressive {
		cloned = truncateLongToolOutputs(cloned, &out, 2048)
	}

	out.Messages = cloned
	out.NewLen = lenOf(cloned)
	out.BytesSaved = max(out.OriginalLen-out.NewLen, 0)
	return out
}

var wsRE = regexp.MustCompile(`[ \t]+`)
var nlRE = regexp.MustCompile(`\n{3,}`)

func collapseWhitespace(msgs []types.Message, r *Result) []types.Message {
	applied := false
	for i, m := range msgs {
		text, ok := decodeContent(m.Content)
		if !ok {
			continue
		}
		new := wsRE.ReplaceAllString(text, " ")
		new = nlRE.ReplaceAllString(new, "\n\n")
		new = strings.TrimSpace(new)
		if new != text {
			applied = true
			msgs[i].Content = encodeContent(new)
		}
	}
	if applied {
		r.Applied = append(r.Applied, "whitespace")
	}
	return msgs
}

func dedupRepeatedToolOutputs(msgs []types.Message, r *Result) []types.Message {
	seen := make(map[string]bool)
	out := make([]types.Message, 0, len(msgs))
	skipped := false
	for _, m := range msgs {
		if m.Role != "tool" {
			out = append(out, m)
			continue
		}
		text, ok := decodeContent(m.Content)
		if !ok {
			out = append(out, m)
			continue
		}
		head := text
		if len(head) > 256 {
			head = head[:256]
		}
		if seen[head] {
			skipped = true
			continue
		}
		seen[head] = true
		out = append(out, m)
	}
	if skipped {
		r.Applied = append(r.Applied, "dedup_tool_outputs")
	}
	return out
}

func truncateLongToolOutputs(msgs []types.Message, r *Result, maxBytes int) []types.Message {
	truncated := false
	for i, m := range msgs {
		if m.Role != "tool" {
			continue
		}
		text, ok := decodeContent(m.Content)
		if !ok || len(text) <= maxBytes {
			continue
		}
		head := text[:maxBytes/2]
		tail := text[len(text)-maxBytes/2:]
		new := head + "\n\n…[truncated " +
			intToStr(len(text)-maxBytes) + " bytes]…\n\n" + tail
		msgs[i].Content = encodeContent(new)
		truncated = true
	}
	if truncated {
		r.Applied = append(r.Applied, "truncate_long_tool")
	}
	return msgs
}

func decodeContent(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s, true
	}
	return "", false
}

func encodeContent(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

func cloneMessages(msgs []types.Message) []types.Message {
	out := make([]types.Message, len(msgs))
	copy(out, msgs)
	return out
}

func lenOf(msgs []types.Message) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content)
	}
	return n
}

func intToStr(i int) string {
	const digits = "0123456789"
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	out := make([]byte, 0, 12)
	for i > 0 {
		out = append([]byte{digits[i%10]}, out...)
		i /= 10
	}
	if neg {
		out = append([]byte{'-'}, out...)
	}
	return string(out)
}
