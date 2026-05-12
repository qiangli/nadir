//go:build onnx

package embed

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// embedder wraps a single ONNX session + tokenizer. It is safe to
// share across goroutines: the tokenizer is read-only and ORT
// AdvancedSession serialises Run() internally.
type embedder struct {
	tok      *Tokenizer
	session  *ort.DynamicAdvancedSession
	maxLen   int
	dim      int
	closeMu  sync.Mutex
	closed   bool
}

const embeddingDim = 384

func openONNX(assetsDir string) (Embedder, error) {
	if err := ensureORT(); err != nil {
		return nil, err
	}
	tok, err := LoadTokenizer(assetsDir)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}
	modelPath := filepath.Join(assetsDir, "model.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("model.onnx not found at %s: %w", modelPath, err)
	}
	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		nil, // default session options
	)
	if err != nil {
		return nil, fmt.Errorf("ort session: %w", err)
	}
	return &embedder{tok: tok, session: sess, maxLen: tok.MaxLen(), dim: embeddingDim}, nil
}

func (e *embedder) Dim() int { return e.dim }

func (e *embedder) Close() error {
	e.closeMu.Lock()
	defer e.closeMu.Unlock()
	if e.closed {
		return nil
	}
	e.closed = true
	if e.session != nil {
		return e.session.Destroy()
	}
	return nil
}

// Embed tokenizes the prompt, runs the ONNX forward pass, mean-pools
// over the sequence dimension using the attention mask, and L2
// normalizes. The output exactly matches the Python pipeline in
// tools/export-onnx/export.py.
func (e *embedder) Embed(_ context.Context, text string) ([]float32, error) {
	ids, mask, types := e.tok.Encode(text)

	inIDs, err := ort.NewTensor(ort.NewShape(1, int64(e.maxLen)), ids)
	if err != nil {
		return nil, fmt.Errorf("ids tensor: %w", err)
	}
	defer inIDs.Destroy()
	inMask, err := ort.NewTensor(ort.NewShape(1, int64(e.maxLen)), mask)
	if err != nil {
		return nil, fmt.Errorf("mask tensor: %w", err)
	}
	defer inMask.Destroy()
	inTypes, err := ort.NewTensor(ort.NewShape(1, int64(e.maxLen)), types)
	if err != nil {
		return nil, fmt.Errorf("types tensor: %w", err)
	}
	defer inTypes.Destroy()

	outShape := ort.NewShape(1, int64(e.maxLen), int64(e.dim))
	out, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, fmt.Errorf("alloc output: %w", err)
	}
	defer out.Destroy()

	if err := e.session.Run(
		[]ort.Value{inIDs, inMask, inTypes},
		[]ort.Value{out},
	); err != nil {
		return nil, fmt.Errorf("ort run: %w", err)
	}

	hidden := out.GetData() // []float32 of length maxLen*dim
	return meanPoolNormalize(hidden, mask, e.maxLen, e.dim), nil
}

// meanPoolNormalize is the Go counterpart of the Python pooling: sum
// hidden states gated by mask, divide by mask count, then L2
// normalize. Matches export.py's embed_one() exactly.
func meanPoolNormalize(hidden []float32, mask []int64, seq, dim int) []float32 {
	out := make([]float32, dim)
	var maskSum float64
	for t := 0; t < seq; t++ {
		if mask[t] == 0 {
			continue
		}
		maskSum++
		offset := t * dim
		for d := 0; d < dim; d++ {
			out[d] += hidden[offset+d]
		}
	}
	if maskSum < 1 {
		maskSum = 1
	}
	for d := 0; d < dim; d++ {
		out[d] /= float32(maskSum)
	}
	var norm float64
	for _, v := range out {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm < 1e-9 {
		norm = 1e-9
	}
	for d := 0; d < dim; d++ {
		out[d] = float32(float64(out[d]) / norm)
	}
	return out
}

// ensureORT performs the one-time global ORT environment init. The
// shared library path can be overridden via NADIR_ONNXRUNTIME_PATH;
// otherwise ORT's default dlopen search applies.
var (
	ortInitOnce sync.Once
	ortInitErr  error
)

func ensureORT() error {
	ortInitOnce.Do(func() {
		if p := os.Getenv("NADIR_ONNXRUNTIME_PATH"); p != "" {
			ort.SetSharedLibraryPath(p)
		}
		if err := ort.InitializeEnvironment(); err != nil {
			ortInitErr = fmt.Errorf("init onnxruntime (set NADIR_ONNXRUNTIME_PATH or install libonnxruntime): %w", err)
		}
	})
	return ortInitErr
}
