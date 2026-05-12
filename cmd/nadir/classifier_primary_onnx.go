//go:build onnx

package main

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/qiangli/nadir/internal/classifier"
	"github.com/qiangli/nadir/internal/types"
)

// buildPrimary (-tags onnx build) requires ONNX. There is no silent
// fallback to heuristic — if the assets or libonnxruntime aren't
// usable, the proxy refuses to start. The intent is: production
// routing must use the production classifier, full stop.
//
// To run with heuristic, build without `-tags onnx`.
func buildPrimary(logger *slog.Logger, thresh classifier.Thresholds) (types.Classifier, string, error) {
	assetsDir := os.Getenv("NADIR_ASSETS_DIR")
	if assetsDir == "" {
		assetsDir = "assets"
	}
	modelPath := filepath.Join(assetsDir, "model.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		return nil, "", fmt.Errorf(
			"ONNX build requires assets at %s, but %s is missing: %w\n"+
				"Run `python tools/export-onnx/export.py` to generate them, "+
				"or set NADIR_ASSETS_DIR to point at an existing assets directory.",
			assetsDir, modelPath, err)
	}
	cls, err := classifier.NewONNXFromAssets(assetsDir, thresh)
	if err != nil {
		return nil, "", fmt.Errorf(
			"ONNX classifier failed to initialize: %w\n"+
				"Verify libonnxruntime is installed (brew install onnxruntime on macOS, "+
				"or set NADIR_ONNXRUNTIME_PATH to the .dylib/.so file).",
			err)
	}
	logger.Info("classifier primary: onnx", slog.String("assets", assetsDir))
	return cls, "onnx", nil
}
