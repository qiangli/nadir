# export-onnx

One-shot Python script that produces the assets the Go ONNX
classifier needs:

```
assets/model.onnx              # all-MiniLM-L6-v2, ~22 MB
assets/vocab.txt               # BERT WordPiece vocab (~30k lines)
assets/tokenizer_config.json   # CLS/SEP/PAD/UNK/max_len
assets/simple_centroid.bin     # 384 float32 LE
assets/complex_centroid.bin    # 384 float32 LE
testdata/classifier_golden.json # 50-prompt cross-language parity fixtures
```

## Usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r tools/export-onnx/requirements.txt
python tools/export-onnx/export.py
```

The script reads `priorart/NadirClaw/nadirclaw/prototypes.py` for the
seed prompts that define the centroids. Re-running the script is
idempotent (modulo PyTorch nondeterminism); the goldens regenerate
in lock-step with the centroids.

## Why re-derive centroids?

The upstream NadirClaw `.npy` files were pooled through
`sentence-transformers`' Python code. The Go path tokenizes + runs the
ONNX export + pools in Go. Even tiny differences in tokenizer
behaviour or pooling op order produce centroid drift. Re-deriving
centroids using the same ONNX path the Go code will run guarantees
the goldens match.

## Build with ONNX

The Go binary defaults to the heuristic classifier. To use ONNX:

```bash
go build -tags onnx -o nadir ./cmd/nadir
```

That pulls in `github.com/yalue/onnxruntime_go` which `dlopen`s the
onnxruntime shared library at runtime. Install onnxruntime locally:

```bash
# macOS
brew install onnxruntime
# Linux
sudo apt install libonnxruntime-dev   # or download from upstream
```

Then point nadir at it (optional — defaults search standard paths):

```bash
NADIR_ONNXRUNTIME_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
    ./nadir serve
```
