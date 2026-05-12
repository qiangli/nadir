package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/qiangli/nadir/types"
)

func TestClientCompleteSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Errorf("auth header = %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(types.ChatResponse{
			ID:    "r-1",
			Model: "gpt-test",
			Choices: []types.Choice{{
				Index:   0,
				Message: types.Message{Role: "assistant", Content: json.RawMessage(`"hi"`)},
			}},
		})
	}))
	defer srv.Close()

	c := New("test", srv.URL, "sk-test")
	resp, err := c.Complete(context.Background(), &types.ChatRequest{Model: "gpt-test", Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}}})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.ID != "r-1" {
		t.Errorf("response ID = %q", resp.ID)
	}
}

func TestClientCompleteRateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(429)
		_, _ = io.WriteString(w, "rate limited")
	}))
	defer srv.Close()

	c := New("test", srv.URL, "sk-test")
	_, err := c.Complete(context.Background(), &types.ChatRequest{Model: "gpt-test"})
	var pe *types.ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("want *ProviderError, got %T: %v", err, err)
	}
	if pe.Kind != types.ErrRateLimit {
		t.Errorf("kind = %s, want rate_limit", pe.Kind)
	}
	if !types.IsTransient(err) {
		t.Errorf("want transient")
	}
}

func TestClientCompleteAuthFatal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(401)
	}))
	defer srv.Close()

	c := New("test", srv.URL, "sk-test")
	_, err := c.Complete(context.Background(), &types.ChatRequest{Model: "gpt-test"})
	if types.IsTransient(err) {
		t.Errorf("401 must be fatal, got transient")
	}
}

func TestClientStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		f := w.(http.Flusher)
		for _, content := range []string{"hel", "lo"} {
			chunk := types.StreamChunk{Model: "gpt-test", Choices: []types.StreamChoice{{Delta: types.StreamDelta{Content: content}}}}
			b, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", b)
			f.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		f.Flush()
	}))
	defer srv.Close()

	c := New("test", srv.URL, "sk-test")
	iter, err := c.Stream(context.Background(), &types.ChatRequest{Model: "gpt-test", Stream: true})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	defer iter.Close()

	var got strings.Builder
	for {
		chunk, err := iter.Next(context.Background())
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("next: %v", err)
		}
		for _, ch := range chunk.Choices {
			got.WriteString(ch.Delta.Content)
		}
	}
	if got.String() != "hello" {
		t.Errorf("stream content = %q, want %q", got.String(), "hello")
	}
}
