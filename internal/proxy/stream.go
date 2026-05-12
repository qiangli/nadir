package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/qiangli/nadir/types"
)

// streamWithFallback runs the streaming fallback loop. The semantics
// match the Python _stream_with_fallback in
// priorart/NadirClaw/nadirclaw/server.py: try each model until the
// first chunk arrives. Once the first byte is on the wire (SSE
// framing committed), no more fallback is possible — mid-stream
// errors surface as an SSE `error` event and the connection closes.
func (s *Server) streamWithFallback(ctx context.Context, w http.ResponseWriter, req *types.ChatRequest, decision *types.RouteDecision, chain []string) {
	_ = decision

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported by writer")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")

	var lastErr error
	for _, model := range chain {
		if !s.deps.Health.Available(model) {
			lastErr = &types.ProviderError{Kind: types.ErrRateLimit, Provider: "health", Model: model, Msg: "model in cooldown"}
			continue
		}
		if s.deps.ModelLimiter != nil {
			if _, ok := s.deps.ModelLimiter.Check(model); !ok {
				lastErr = &types.ProviderError{Kind: types.ErrRateLimit, Provider: "ratelimit", Model: model, Msg: "model in cooldown"}
				continue
			}
		}
		client, err := s.pickProvider(model)
		if err != nil {
			lastErr = err
			if !types.IsTransient(err) {
				writeStreamError(w, flusher, err)
				return
			}
			continue
		}

		clone := *req
		clone.Model = model
		clone.Stream = true

		iter, err := client.Stream(ctx, &clone)
		if err != nil {
			s.recordFailure(model, err)
			lastErr = err
			if !types.IsTransient(err) {
				writeStreamError(w, flusher, err)
				return
			}
			continue
		}

		// Probe the first chunk before committing the stream. If it
		// errors before yielding, we can still fall back.
		first, err := iter.Next(ctx)
		if err != nil {
			iter.Close()
			if errors.Is(err, io.EOF) {
				// Empty stream — treat as a transient hiccup.
				s.recordFailure(model, &types.ProviderError{Kind: types.ErrServerError, Provider: client.Name(), Model: model, Msg: "empty stream"})
				continue
			}
			s.recordFailure(model, err)
			lastErr = err
			if !types.IsTransient(err) {
				writeStreamError(w, flusher, err)
				return
			}
			continue
		}

		// First chunk in hand — commit to this model.
		writeStreamChunk(w, flusher, first)
		for {
			chunk, err := iter.Next(ctx)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				writeStreamError(w, flusher, err)
				iter.Close()
				s.recordFailure(model, err)
				return
			}
			writeStreamChunk(w, flusher, chunk)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		iter.Close()
		s.deps.Health.RecordSuccess(model)
		return
	}

	// Whole chain failed before the first chunk.
	if lastErr == nil {
		lastErr = errors.New("no models in fallback chain")
	}
	writeStreamError(w, flusher, lastErr)
}

func writeStreamChunk(w http.ResponseWriter, flusher http.Flusher, chunk *types.StreamChunk) {
	b, err := json.Marshal(chunk)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", b)
	flusher.Flush()
}

func writeStreamError(w http.ResponseWriter, flusher http.Flusher, err error) {
	payload := map[string]any{
		"error": map[string]any{
			"message": err.Error(),
		},
	}
	b, _ := json.Marshal(payload)
	fmt.Fprintf(w, "event: error\ndata: %s\n\n", b)
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}
