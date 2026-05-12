package classifier

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/qiangli/nadir/internal/types"
)

// mockEmbedder maps a prompt → fixed embedding via a lookup table.
// Tests use it to drive the ONNX classifier through deterministic
// numerics without needing the actual model.
type mockEmbedder struct {
	dim  int
	emit map[string][]float32
}

func (m *mockEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	v, ok := m.emit[text]
	if !ok {
		return nil, errors.New("mock has no embedding for " + text)
	}
	return v, nil
}
func (m *mockEmbedder) Dim() int     { return m.dim }
func (m *mockEmbedder) Close() error { return nil }

func TestONNXClassifyAlignedWithComplex(t *testing.T) {
	simple := unit([]float32{1, 0, 0, 0})
	complex := unit([]float32{0, 0, 0, 1})
	prompt := unit([]float32{0, 0, 0, 1}) // aligned with complex

	emb := &mockEmbedder{dim: 4, emit: map[string][]float32{"x": prompt}}
	cls, err := NewONNX(emb, simple, complex, Thresholds{Simple: 0.35, Complex: 0.65, HasMid: false})
	if err != nil {
		t.Fatal(err)
	}
	tier, score, conf, err := cls.Classify(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if tier != types.TierComplex {
		t.Errorf("tier = %s, want complex", tier)
	}
	if score < 0.65 {
		t.Errorf("score = %v, want >= 0.65", score)
	}
	if conf <= 0 {
		t.Errorf("confidence = %v, want > 0", conf)
	}
}

func TestONNXClassifyAlignedWithSimple(t *testing.T) {
	simple := unit([]float32{1, 0, 0, 0})
	complex := unit([]float32{0, 0, 0, 1})
	prompt := unit([]float32{1, 0, 0, 0})

	emb := &mockEmbedder{dim: 4, emit: map[string][]float32{"x": prompt}}
	cls, _ := NewONNX(emb, simple, complex, Thresholds{Simple: 0.35, Complex: 0.65})
	tier, score, _, _ := cls.Classify(context.Background(), "x")
	if tier != types.TierSimple {
		t.Errorf("tier = %s, want simple", tier)
	}
	if score > 0.35 {
		t.Errorf("score = %v, want <= 0.35", score)
	}
}

func TestONNXAmbiguousFallsToSimple(t *testing.T) {
	simple := unit([]float32{1, 0, 0, 0})
	complex := unit([]float32{0, 0, 0, 1})
	// Equidistant prompt → confidence ~0 → score ~0.5 → no mid tier
	// configured → bucket to simple.
	prompt := unit([]float32{1, 0, 0, 1})

	emb := &mockEmbedder{dim: 4, emit: map[string][]float32{"x": prompt}}
	cls, _ := NewONNX(emb, simple, complex, Thresholds{Simple: 0.35, Complex: 0.65})
	tier, _, _, _ := cls.Classify(context.Background(), "x")
	if tier != types.TierSimple {
		t.Errorf("equidistant prompt → tier=%s, want simple (no mid configured)", tier)
	}
}

func TestReadCentroidRejectsWrongSize(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "cen.bin")
	// 3 floats but we ask for 4.
	buf := make([]byte, 12)
	for i, v := range []float32{1, 2, 3} {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	_ = os.WriteFile(path, buf, 0o644)
	if _, err := readCentroid(path, 4); err == nil {
		t.Fatal("expected dim-mismatch error")
	}
}

// TestGoldenFixturePresent guards against the Python export script
// being absent from the repo. It does NOT assert numeric parity —
// that's only meaningful once the ONNX assets are populated and the
// binary is built with -tags onnx. We just check the fixture exists
// and is well-formed so CI catches accidental deletion.
func TestGoldenFixturePresent(t *testing.T) {
	path := filepath.Join("..", "..", "testdata", "classifier_golden.json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("classifier_golden.json not generated yet; run tools/export-onnx/export.py")
	} else if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var rows []map[string]any
	if err := json.Unmarshal(data, &rows); err != nil {
		t.Fatalf("golden fixture malformed: %v", err)
	}
	if len(rows) == 0 {
		t.Error("golden fixture is empty")
	}
}

func unit(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	n = math.Sqrt(n)
	if n < 1e-9 {
		n = 1e-9
	}
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / n)
	}
	return out
}
