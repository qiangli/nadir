// Package types defines the wire and internal types shared across nadir
// packages. The chat request/response shapes mirror the OpenAI
// /v1/chat/completions JSON envelope so nadir can act as a drop-in
// proxy.
package types

import (
	"encoding/json"
	"time"
)

type Tier string

const (
	TierSimple    Tier = "simple"
	TierMid       Tier = "mid"
	TierComplex   Tier = "complex"
	TierReasoning Tier = "reasoning"
	TierFree      Tier = "free"
)

// Message is one entry in a chat conversation. Content is either a
// plain string or a list of content parts (OpenAI vision/tool shape).
type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	Name       string          `json:"name,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Tool struct {
	Type     string          `json:"type"`
	Function json.RawMessage `json:"function"`
}

// ChatRequest is the OpenAI /v1/chat/completions request body. Unknown
// fields pass through via Extra so we don't drop provider-specific
// hints (Anthropic cache_control, Gemini system_instruction, etc.).
type ChatRequest struct {
	Model            string                 `json:"model"`
	Messages         []Message              `json:"messages"`
	Stream           bool                   `json:"stream,omitempty"`
	Temperature      *float64               `json:"temperature,omitempty"`
	TopP             *float64               `json:"top_p,omitempty"`
	MaxTokens        *int                   `json:"max_tokens,omitempty"`
	N                *int                   `json:"n,omitempty"`
	Stop             json.RawMessage        `json:"stop,omitempty"`
	PresencePenalty  *float64               `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64               `json:"frequency_penalty,omitempty"`
	User             string                 `json:"user,omitempty"`
	Tools            []Tool                 `json:"tools,omitempty"`
	ToolChoice       json.RawMessage        `json:"tool_choice,omitempty"`
	ResponseFormat   json.RawMessage        `json:"response_format,omitempty"`
	Extra            map[string]any         `json:"-"`
}

// ChatResponse is the OpenAI /v1/chat/completions non-streaming response.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamChunk mirrors an OpenAI streaming SSE event. Done=true signals
// the [DONE] sentinel; iterators may emit a final chunk with Usage set
// when the upstream provides it.
type StreamChunk struct {
	ID      string         `json:"id,omitempty"`
	Object  string         `json:"object,omitempty"`
	Created int64          `json:"created,omitempty"`
	Model   string         `json:"model,omitempty"`
	Choices []StreamChoice `json:"choices,omitempty"`
	Usage   *Usage         `json:"usage,omitempty"`
	Done    bool           `json:"-"`
}

type StreamChoice struct {
	Index        int            `json:"index"`
	Delta        StreamDelta    `json:"delta"`
	FinishReason string         `json:"finish_reason,omitempty"`
}

type StreamDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// RouteDecision is the router's output: the model to call, the
// fallback chain, the classifier score, and any modifiers applied.
type RouteDecision struct {
	Model         string
	Tier          Tier
	Provider      string
	Score         float64
	Confidence    float64
	Cached        bool
	Modifiers     []string
	FallbackChain []string
}

// RequestMeta is the classifier/router input derived from a ChatRequest:
// the user-visible text, image/tool flags, and timing data the router
// uses for modifier decisions.
type RequestMeta struct {
	UserText    string
	SystemText  string
	HasImages   bool
	HasTools    bool
	ToolNames   []string
	MessageCount int
	TotalChars  int
	Received    time.Time
}

// User identifies the caller for rate-limiting, budget tracking, and
// session-cache pinning. Anonymous requests get a synthetic ID derived
// from the remote address.
type User struct {
	ID    string
	Token string
}
