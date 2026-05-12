//go:build !onnx

package main

import (
	"log/slog"
	"os"
	"path/filepath"

	"github.com/qiangli/nadir/classifier"
	"github.com/qiangli/nadir/types"
)

// buildPrimary (default build) returns the heuristic classifier.
// There is no production-grade primary in this build path because
// the ONNX subsystem wasn't compiled in — that's the contract of
// the default build tag.
//
// We log loudly when ONNX assets *are* present, since that's a
// signal the operator probably meant to build `-tags onnx` and
// forgot.
func buildPrimary(logger *slog.Logger, thresh classifier.Thresholds) (types.Classifier, string, error) {
	assetsDir := os.Getenv("NADIR_ASSETS_DIR")
	if assetsDir == "" {
		assetsDir = "assets"
	}
	if _, err := os.Stat(filepath.Join(assetsDir, "model.onnx")); err == nil {
		logger.Warn("ONNX assets present but binary was NOT compiled with -tags onnx — using heuristic anyway. Rebuild with `-tags onnx` for production routing.",
			slog.String("assets_dir", assetsDir),
		)
	} else {
		logger.Info("classifier primary: heuristic (development / no-ONNX build)",
			slog.String("assets_dir", assetsDir),
		)
	}
	return classifier.NewHeuristic(thresh), "heuristic", nil
}
