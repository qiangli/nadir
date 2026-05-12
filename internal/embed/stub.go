//go:build !onnx

package embed

// openONNX in the default build returns ErrNotEnabled so callers get
// a clear "rebuild with -tags onnx" message instead of a cryptic
// runtime crash. The Real™ implementation lives in onnx.go.
func openONNX(_ string) (Embedder, error) {
	return nil, ErrNotEnabled
}
