//go:build onnx

package classifier

// Numerical-parity test that compares our ONNX-backed classifier
// against the Python pipeline. Both run the same MiniLM-L6-v2 model
// against the same prompts; the goldens were produced by
// tools/export-onnx/export.py (which also writes the model + vocab +
// centroids in assets/).
//
// Build + run:
//
//   export NADIR_ONNXRUNTIME_PATH=/opt/homebrew/lib/libonnxruntime.dylib
//   go test -tags onnx -run TestONNXGolden -v ./classifier/

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type goldenRow struct {
	Prompt     string  `json:"prompt"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
	Tier       string  `json:"tier"`
	SimSimple  float64 `json:"sim_simple"`
	SimComplex float64 `json:"sim_complex"`
}

func TestONNXGoldenNumericalParity(t *testing.T) {
	assetsDir := filepath.Join("..", "..", "assets")
	if _, err := os.Stat(filepath.Join(assetsDir, "model.onnx")); err != nil {
		t.Skipf("assets/model.onnx missing — run python tools/export-onnx/export.py first: %v", err)
	}
	goldenPath := filepath.Join("..", "..", "testdata", "classifier_golden.json")
	data, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Skipf("classifier_golden.json missing: %v", err)
	}
	var rows []goldenRow
	if err := json.Unmarshal(data, &rows); err != nil {
		t.Fatalf("malformed golden: %v", err)
	}

	cls, err := NewONNXFromAssets(assetsDir, DefaultThresholds())
	if err != nil {
		t.Fatalf("load ONNX classifier: %v", err)
	}
	defer cls.Close()

	tolScore := 1e-4
	tolSim := 1e-4
	scoreMatch, simMatch, tierMatch := 0, 0, 0
	for i, row := range rows {
		_, score, conf, err := cls.Classify(context.Background(), row.Prompt)
		if err != nil {
			t.Fatalf("classify %d: %v", i, err)
		}
		// Recompute sim_simple/sim_complex from emb + centroids so we
		// can compare raw similarity numerics.
		emb, err := cls.emb.Embed(context.Background(), row.Prompt)
		if err != nil {
			t.Fatalf("embed %d: %v", i, err)
		}
		simS := dot(emb, cls.simpleCen)
		simC := dot(emb, cls.complexCen)

		ds := math.Abs(score - row.Score)
		dsS := math.Abs(simS - row.SimSimple)
		dsC := math.Abs(simC - row.SimComplex)

		if ds <= tolScore {
			scoreMatch++
		} else {
			t.Logf("row %d %q: score Δ=%.6f (go=%.6f py=%.6f)", i, truncStr(row.Prompt, 40), ds, score, row.Score)
		}
		if dsS <= tolSim && dsC <= tolSim {
			simMatch++
		} else {
			t.Logf("row %d %q: sim Δsimple=%.6f Δcomplex=%.6f (go=%.6f,%.6f py=%.6f,%.6f)",
				i, truncStr(row.Prompt, 40), dsS, dsC, simS, simC, row.SimSimple, row.SimComplex)
		}

		// Tier comparison — our local heuristic mid bucketing differs
		// from a stable two-tier (simple/complex) split when mid is
		// disabled. The golden uses no-mid thresholds 0.35/0.65 so
		// scores in (0.35, 0.65) bucket to simple here.
		expectedTier := row.Tier
		gotTier, _ := bucketTier(score)
		if string(gotTier) == expectedTier {
			tierMatch++
		}
		_ = conf // currently only used for tier mapping above
	}
	t.Logf("score parity:      %d/%d", scoreMatch, len(rows))
	t.Logf("similarity parity: %d/%d", simMatch, len(rows))
	t.Logf("tier parity:       %d/%d", tierMatch, len(rows))

	if simMatch < len(rows) {
		t.Errorf("similarity parity: %d/%d below 100%%", simMatch, len(rows))
	}
	if scoreMatch < len(rows) {
		t.Errorf("score parity: %d/%d below 100%%", scoreMatch, len(rows))
	}
}

func truncStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func bucketTier(score float64) (tier string, ok bool) {
	if score <= 0.35 {
		return "simple", true
	}
	if score >= 0.65 {
		return "complex", true
	}
	return "simple", true // no mid by default in goldens
}
