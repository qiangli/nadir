package embed

import (
	"context"
	"errors"
)

// ErrNotEnabled is returned when ONNX support was not compiled into
// the binary. Build with `-tags onnx` to enable it.
var ErrNotEnabled = errors.New("nadir: ONNX support not compiled in (rebuild with -tags onnx)")

// Embedder turns a prompt into a 384-dim L2-normalized embedding.
// Implementations are stateful (they own a tokenizer + ONNX session)
// and must be initialised once and reused; Close releases the
// underlying ORT resources.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Dim() int
	Close() error
}

// Open returns an ONNX-backed embedder built from the assets at
// assetsDir/{model.onnx,vocab.txt,tokenizer_config.json}. In the
// default build it returns ErrNotEnabled — the real implementation
// lives in onnx.go behind the `onnx` build tag.
func Open(assetsDir string) (Embedder, error) {
	return openONNX(assetsDir)
}
