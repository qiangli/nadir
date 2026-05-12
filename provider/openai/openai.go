// Package openai is the OpenAI-compatible LLMClient. It targets the
// /v1/chat/completions endpoint shape so the same code can drive
// OpenAI, Ollama (`/v1` mode), vLLM, LocalAI, and any other server
// that speaks the standard wire format — choose the right server by
// varying BaseURL.
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"strings"
	"time"

	"github.com/qiangli/nadir/types"
)

const defaultBaseURL = "https://api.openai.com/v1"

type Client struct {
	BaseURL string
	APIKey  string
	HTTP    *http.Client
	NameStr string
	OrgID   string
}

func New(name, baseURL, apiKey string) *Client {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		APIKey:  apiKey,
		HTTP:    &http.Client{Timeout: 5 * time.Minute},
		NameStr: name,
	}
}

func (c *Client) Name() string { return c.NameStr }

func (c *Client) Complete(ctx context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	body, err := marshalRequest(req, false)
	if err != nil {
		return nil, &types.ProviderError{Kind: types.ErrValidation, Provider: c.NameStr, Model: req.Model, Msg: err.Error()}
	}
	httpReq, err := c.newRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return nil, c.networkError(req.Model, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, c.httpError(req.Model, resp)
	}
	var out types.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, &types.ProviderError{Kind: types.ErrServerError, Provider: c.NameStr, Model: req.Model, Msg: "decode: " + err.Error()}
	}
	return &out, nil
}

func (c *Client) Stream(ctx context.Context, req *types.ChatRequest) (types.StreamIter, error) {
	body, err := marshalRequest(req, true)
	if err != nil {
		return nil, &types.ProviderError{Kind: types.ErrValidation, Provider: c.NameStr, Model: req.Model, Msg: err.Error()}
	}
	httpReq, err := c.newRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Accept", "text/event-stream")
	resp, err := c.HTTP.Do(httpReq)
	if err != nil {
		return nil, c.networkError(req.Model, err)
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		return nil, c.httpError(req.Model, resp)
	}
	return &streamIter{
		provider: c.NameStr,
		model:    req.Model,
		resp:     resp,
		reader:   bufio.NewReaderSize(resp.Body, 64*1024),
	}, nil
}

func (c *Client) newRequest(ctx context.Context, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, &types.ProviderError{Kind: types.ErrUnknown, Provider: c.NameStr, Msg: err.Error()}
	}
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	if c.OrgID != "" {
		req.Header.Set("OpenAI-Organization", c.OrgID)
	}
	return req, nil
}

func (c *Client) httpError(model string, resp *http.Response) error {
	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	kind := classifyStatus(resp.StatusCode)
	return &types.ProviderError{
		Kind:       kind,
		StatusCode: resp.StatusCode,
		Provider:   c.NameStr,
		Model:      model,
		Msg:        strings.TrimSpace(string(bodyBytes)),
	}
}

func (c *Client) networkError(model string, err error) error {
	kind := types.ErrNetwork
	if errors.Is(err, context.DeadlineExceeded) {
		kind = types.ErrTimeout
	}
	if errors.Is(err, context.Canceled) {
		kind = types.ErrCanceled
	}
	return &types.ProviderError{
		Kind:       kind,
		Provider:   c.NameStr,
		Model:      model,
		Msg:        err.Error(),
		Underlying: err,
	}
}

func classifyStatus(code int) types.ErrorKind {
	switch {
	case code == 401 || code == 403:
		return types.ErrAuth
	case code == 408:
		return types.ErrTimeout
	case code == 429:
		return types.ErrRateLimit
	case code >= 500:
		return types.ErrServerError
	case code >= 400:
		return types.ErrBadRequest
	default:
		return types.ErrUnknown
	}
}

// marshalRequest is the one place we re-encode the ChatRequest so we
// can flip the stream flag and drop nadir-internal-only fields. Extra
// passthrough fields are merged back into the JSON object.
func marshalRequest(req *types.ChatRequest, stream bool) ([]byte, error) {
	clone := *req
	clone.Stream = stream
	out, err := json.Marshal(&clone)
	if err != nil {
		return nil, err
	}
	if len(req.Extra) == 0 {
		return out, nil
	}
	var generic map[string]any
	if err := json.Unmarshal(out, &generic); err != nil {
		return nil, err
	}
	maps.Copy(generic, req.Extra)
	return json.Marshal(generic)
}

// streamIter consumes an SSE response body. Each "data: <json>" line is
// decoded as a StreamChunk; the [DONE] sentinel returns io.EOF.
type streamIter struct {
	provider string
	model    string
	resp     *http.Response
	reader   *bufio.Reader
	closed   bool
}

func (s *streamIter) Next(ctx context.Context) (*types.StreamChunk, error) {
	if s.closed {
		return nil, io.EOF
	}
	for {
		if err := ctx.Err(); err != nil {
			return nil, &types.ProviderError{Kind: types.ErrCanceled, Provider: s.provider, Model: s.model, Msg: err.Error()}
		}
		line, err := s.reader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil, io.EOF
			}
			return nil, &types.ProviderError{Kind: types.ErrNetwork, Provider: s.provider, Model: s.model, Msg: err.Error(), Underlying: err}
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			return nil, io.EOF
		}
		var chunk types.StreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			return nil, &types.ProviderError{Kind: types.ErrServerError, Provider: s.provider, Model: s.model, Msg: "sse decode: " + err.Error()}
		}
		return &chunk, nil
	}
}

func (s *streamIter) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	return s.resp.Body.Close()
}

var _ types.LLMClient = (*Client)(nil)
