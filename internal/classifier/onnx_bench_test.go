//go:build onnx

package classifier

// Latency benchmark for the ONNX classifier. NadirClaw advertises
// "~10ms classification overhead"; this runs the same pipeline on the
// Go side and reports ns/op so we can confirm we hit that budget.
//
// Run:
//
//   NADIR_ONNXRUNTIME_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
//     go test -tags onnx -bench BenchmarkONNXClassify -benchtime=10s \
//     -run '^$' ./internal/classifier/

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

var benchPrompts = []string{
	"What is 2+2?",
	"How do I exit vim?",
	"Refactor this concurrent code to eliminate the race condition and explain why the original deadlocks step by step.",
	"Design a distributed event-sourcing system for a financial trading platform with consensus protocol.",
	"hi",
}

func BenchmarkONNXClassify(b *testing.B) {
	assetsDir := filepath.Join("..", "..", "assets")
	if _, err := os.Stat(filepath.Join(assetsDir, "model.onnx")); err != nil {
		b.Skipf("assets missing: %v", err)
	}
	cls, err := NewONNXFromAssets(assetsDir, DefaultThresholds())
	if err != nil {
		b.Fatal(err)
	}
	defer cls.Close()

	// Warmup so the first ORT Run doesn't skew the average.
	_ = cls.Warmup(context.Background())

	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, err := cls.Classify(ctx, benchPrompts[i%len(benchPrompts)])
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkONNXClassifyShortPrompt(b *testing.B) {
	assetsDir := filepath.Join("..", "..", "assets")
	if _, err := os.Stat(filepath.Join(assetsDir, "model.onnx")); err != nil {
		b.Skipf("assets missing: %v", err)
	}
	cls, _ := NewONNXFromAssets(assetsDir, DefaultThresholds())
	defer cls.Close()
	_ = cls.Warmup(context.Background())

	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _, _ = cls.Classify(ctx, "hi")
	}
}
