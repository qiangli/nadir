// Package fake contains LLMClient implementations used in tests:
// FakeOK returns canned content, Fake429/Fake5xx/FakeTimeout return
// the error kinds that drive the fallback loop, FakeStream emits a
// fixed list of chunks, and FakeStreamFailMidway emits N chunks then
// errors so streaming-fallback edge cases can be asserted.
package fake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync/atomic"
	"time"

	"github.com/qiangli/nadir/internal/types"
)

// OK returns a fixed response. Calls counts invocations so tests can
// assert fallback ordering.
type OK struct {
	NameStr string
	Content string
	Calls   atomic.Int64
}

func (f *OK) Name() string { return f.NameStr }

func (f *OK) Complete(_ context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	n := f.Calls.Add(1)
	content, _ := json.Marshal(f.Content)
	return &types.ChatResponse{
		ID:      fmt.Sprintf("fake-%s-%d", f.NameStr, n),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
		Choices: []types.Choice{
			{
				Index:        0,
				Message:      types.Message{Role: "assistant", Content: content},
				FinishReason: "stop",
			},
		},
		Usage: &types.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}, nil
}

func (f *OK) Stream(_ context.Context, req *types.ChatRequest) (types.StreamIter, error) {
	f.Calls.Add(1)
	return &fixedStream{model: req.Model, content: f.Content}, nil
}

// Err is a provider that always returns the configured error kind.
type Err struct {
	NameStr string
	Kind    types.ErrorKind
	Calls   atomic.Int64
}

func (f *Err) Name() string { return f.NameStr }

func (f *Err) err(model string) error {
	return &types.ProviderError{
		Kind:       f.Kind,
		StatusCode: statusFor(f.Kind),
		Provider:   f.NameStr,
		Model:      model,
		Msg:        "fake error",
	}
}

func (f *Err) Complete(_ context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	f.Calls.Add(1)
	return nil, f.err(req.Model)
}

func (f *Err) Stream(_ context.Context, req *types.ChatRequest) (types.StreamIter, error) {
	f.Calls.Add(1)
	return nil, f.err(req.Model)
}

func statusFor(k types.ErrorKind) int {
	switch k {
	case types.ErrRateLimit:
		return 429
	case types.ErrServerError:
		return 503
	case types.ErrTimeout:
		return 504
	case types.ErrAuth:
		return 401
	case types.ErrValidation, types.ErrBadRequest:
		return 400
	default:
		return 500
	}
}

// Timeout sleeps until ctx fires, then returns ErrTimeout. Used to
// validate the per-call deadline in the fallback loop.
type Timeout struct {
	NameStr string
	Delay   time.Duration
}

func (f *Timeout) Name() string { return f.NameStr }

func (f *Timeout) Complete(ctx context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	t := time.NewTimer(f.Delay)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return nil, &types.ProviderError{
			Kind:     types.ErrTimeout,
			Provider: f.NameStr,
			Model:    req.Model,
			Msg:      "ctx deadline",
		}
	case <-t.C:
		return nil, &types.ProviderError{
			Kind:     types.ErrTimeout,
			Provider: f.NameStr,
			Model:    req.Model,
			Msg:      "fake delay elapsed",
		}
	}
}

func (f *Timeout) Stream(ctx context.Context, req *types.ChatRequest) (types.StreamIter, error) {
	_, err := f.Complete(ctx, req)
	return nil, err
}

// fixedStream emits a single chunk with the canned content, then EOF.
type fixedStream struct {
	model   string
	content string
	emitted bool
	done    bool
}

func (s *fixedStream) Next(_ context.Context) (*types.StreamChunk, error) {
	if s.done {
		return nil, io.EOF
	}
	if !s.emitted {
		s.emitted = true
		return &types.StreamChunk{
			ID:      "fake-stream",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   s.model,
			Choices: []types.StreamChoice{
				{Index: 0, Delta: types.StreamDelta{Role: "assistant", Content: s.content}},
			},
		}, nil
	}
	s.done = true
	return &types.StreamChunk{
		ID:    "fake-stream",
		Model: s.model,
		Choices: []types.StreamChoice{
			{Index: 0, FinishReason: "stop"},
		},
		Usage: &types.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}, nil
}

func (s *fixedStream) Close() error { return nil }

// StreamFailMidway emits N chunks then errors. Used to validate that
// once the first byte is committed to the client, the proxy does NOT
// try to fall back to another model.
type StreamFailMidway struct {
	NameStr string
	Chunks  int
	Calls   atomic.Int64
}

func (f *StreamFailMidway) Name() string { return f.NameStr }

func (f *StreamFailMidway) Complete(_ context.Context, _ *types.ChatRequest) (*types.ChatResponse, error) {
	return nil, errors.New("StreamFailMidway: not implemented for non-streaming")
}

func (f *StreamFailMidway) Stream(_ context.Context, req *types.ChatRequest) (types.StreamIter, error) {
	f.Calls.Add(1)
	return &midwayStream{model: req.Model, total: f.Chunks}, nil
}

type midwayStream struct {
	model     string
	total     int
	delivered int
}

func (s *midwayStream) Next(_ context.Context) (*types.StreamChunk, error) {
	if s.delivered >= s.total {
		return nil, &types.ProviderError{
			Kind:     types.ErrServerError,
			Provider: "fake",
			Model:    s.model,
			Msg:      fmt.Sprintf("mid-stream failure after %d chunks", s.delivered),
		}
	}
	s.delivered++
	return &types.StreamChunk{
		Model: s.model,
		Choices: []types.StreamChoice{
			{Index: 0, Delta: types.StreamDelta{Content: "chunk"}},
		},
	}, nil
}

func (s *midwayStream) Close() error { return nil }

var (
	_ types.LLMClient = (*OK)(nil)
	_ types.LLMClient = (*Err)(nil)
	_ types.LLMClient = (*Timeout)(nil)
	_ types.LLMClient = (*StreamFailMidway)(nil)
)
