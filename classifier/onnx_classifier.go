package classifier

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/qiangli/nadir/embed"
	"github.com/qiangli/nadir/types"
)

// Embedder is the seam between this package and github.com/qiangli/nadir/embed.
// Kept local so the classifier doesn't import embed transitively in
// tests where we substitute a mock.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	Dim() int
	Close() error
}

// ONNX is the production classifier: it runs the embedding through
// pre-computed simple/complex centroids and maps the result to a
// tier. Numerics exactly mirror priorart/NadirClaw/.../classifier.py
// and the Python golden fixture at testdata/classifier_golden.json.
type ONNX struct {
	emb        Embedder
	simpleCen  []float32
	complexCen []float32
	thresholds Thresholds
}

// NewONNXFromAssets opens the embedder against assetsDir and loads
// the two centroid blobs. Returns embed.ErrNotEnabled if the binary
// was compiled without `-tags onnx`.
func NewONNXFromAssets(assetsDir string, t Thresholds) (*ONNX, error) {
	emb, err := embed.Open(assetsDir)
	if err != nil {
		return nil, err
	}
	simple, err := readCentroid(filepath.Join(assetsDir, "simple_centroid.bin"), emb.Dim())
	if err != nil {
		_ = emb.Close()
		return nil, fmt.Errorf("simple centroid: %w", err)
	}
	complex, err := readCentroid(filepath.Join(assetsDir, "complex_centroid.bin"), emb.Dim())
	if err != nil {
		_ = emb.Close()
		return nil, fmt.Errorf("complex centroid: %w", err)
	}
	return &ONNX{emb: emb, simpleCen: simple, complexCen: complex, thresholds: t}, nil
}

// NewONNX is the test-friendly constructor that injects the embedder
// and the two centroids directly.
func NewONNX(emb Embedder, simpleCen, complexCen []float32, t Thresholds) (*ONNX, error) {
	if len(simpleCen) != emb.Dim() || len(complexCen) != emb.Dim() {
		return nil, errors.New("centroid dimension mismatch")
	}
	return &ONNX{emb: emb, simpleCen: simpleCen, complexCen: complexCen, thresholds: t}, nil
}

func (o *ONNX) Warmup(ctx context.Context) error {
	_, _, _, err := o.Classify(ctx, "hello")
	return err
}

func (o *ONNX) Classify(ctx context.Context, prompt string) (types.Tier, float64, float64, error) {
	emb, err := o.emb.Embed(ctx, prompt)
	if err != nil {
		return "", 0, 0, err
	}
	simSimple := dot(emb, o.simpleCen)
	simComplex := dot(emb, o.complexCen)
	confidence := math.Abs(simComplex - simSimple)

	scaled := confidence * 5.0
	if scaled > 0.5 {
		scaled = 0.5
	}
	score := 0.5 - scaled
	if simComplex > simSimple {
		score = 0.5 + scaled
	}

	tier := BucketScore(score, o.thresholds)
	return tier, score, confidence, nil
}

func (o *ONNX) Close() error {
	if o.emb != nil {
		return o.emb.Close()
	}
	return nil
}

func dot(a, b []float32) float64 {
	var sum float64
	for i, v := range a {
		sum += float64(v) * float64(b[i])
	}
	return sum
}

func readCentroid(path string, expectedDim int) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) != expectedDim*4 {
		return nil, fmt.Errorf("size %d != expected %d (dim*4)", len(data), expectedDim*4)
	}
	out := make([]float32, expectedDim)
	for i := range expectedDim {
		bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		out[i] = math.Float32frombits(bits)
	}
	return out, nil
}

var _ types.Classifier = (*ONNX)(nil)
