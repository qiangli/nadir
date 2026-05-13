package skillrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OllamaEmbedder calls Ollama's `/api/embeddings` endpoint to produce
// vectors from any model that Ollama exposes as an embedder
// (nomic-embed-text, qwen3-embedding, mxbai-embed-large, etc.). Pure
// Go — stdlib HTTP only, no CGO, no model download by this package
// (Ollama itself handles model storage).
//
// Output is L2-normalized to match the Embedder interface's
// dot-product-is-cosine convention. Ollama returns vectors in raw
// floats; whether they're already normalized depends on the model,
// so we normalize defensively.
//
// Use this when you want MiniLM-quality (or better) semantic
// embeddings without the ONNX/CGO build constraint. Trade-off: every
// embed is an HTTP round-trip — ~5–50 ms depending on model size
// and whether the model is already loaded in Ollama's cache.
type OllamaEmbedder struct {
	baseURL string
	model   string
	dim     int
	client  *http.Client
}

// OllamaOption configures an OllamaEmbedder.
type OllamaOption func(*OllamaEmbedder)

// WithOllamaTimeout caps each /api/embeddings round-trip. Default 30s
// — generous enough for cold-start model loads, tight enough that a
// dead Ollama doesn't hang routing indefinitely.
func WithOllamaTimeout(d time.Duration) OllamaOption {
	return func(e *OllamaEmbedder) {
		if d > 0 {
			e.client.Timeout = d
		}
	}
}

// WithOllamaHTTPClient lets the caller inject a pre-configured client
// (custom Transport for keep-alives, proxies, etc.). Overrides
// WithOllamaTimeout.
func WithOllamaHTTPClient(c *http.Client) OllamaOption {
	return func(e *OllamaEmbedder) {
		if c != nil {
			e.client = c
		}
	}
}

// NewOllamaEmbedder dials Ollama at baseURL, requests a warmup
// embedding to discover the model's dimensionality, and returns a
// ready-to-use Embedder. Returns an error if Ollama is unreachable or
// the model isn't pulled.
//
// baseURL is the bare Ollama URL (e.g., "http://localhost:11434" — no
// "/v1" suffix; this uses the native /api/embeddings, not the
// OpenAI-compatibility shim).
func NewOllamaEmbedder(ctx context.Context, baseURL, model string, opts ...OllamaOption) (*OllamaEmbedder, error) {
	if baseURL == "" {
		return nil, errors.New("skillrouter: empty Ollama baseURL")
	}
	if model == "" {
		return nil, errors.New("skillrouter: empty Ollama model")
	}
	e := &OllamaEmbedder{
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(e)
	}
	// Probe to discover dimension; also surfaces "model not pulled"
	// failures at construction time rather than at first Route.
	vec, err := e.callEmbed(ctx, "warmup")
	if err != nil {
		return nil, fmt.Errorf("skillrouter: ollama probe: %w", err)
	}
	e.dim = len(vec)
	if e.dim == 0 {
		return nil, errors.New("skillrouter: ollama returned 0-dim embedding")
	}
	return e, nil
}

// Embed produces an L2-normalized embedding for the text.
func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vec, err := e.callEmbed(ctx, text)
	if err != nil {
		return nil, err
	}
	l2Normalize(vec)
	return vec, nil
}

// callEmbed is the raw round-trip without normalization.
func (e *OllamaEmbedder) callEmbed(ctx context.Context, text string) ([]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": e.model, "prompt": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, string(b))
	}
	var out struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Embedding) == 0 {
		return nil, errors.New("ollama: empty embedding")
	}
	return out.Embedding, nil
}

// Dim returns the discovered embedding dimension.
func (e *OllamaEmbedder) Dim() int { return e.dim }

// Close is a no-op — http.Client has no shutdown to call.
func (e *OllamaEmbedder) Close() error { return nil }

var _ Embedder = (*OllamaEmbedder)(nil)
