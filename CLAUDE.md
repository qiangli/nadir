# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, test, lint

Use the `Makefile`; don't memorize the underlying flags.

- `make build` — default binary (heuristic classifier, no CGO, no ML deps) → `bin/nadir`
- `make build-onnx` — adds the MiniLM ONNX classifier. Requires CGO + `libonnxruntime` (e.g. `brew install onnxruntime`) at runtime; path overridable via `NADIR_ONNXRUNTIME_PATH`
- `make test` / `make test-onnx` — unit tests on the default vs `-tags onnx` build
- `make tidy` — `gofmt -s -w .` then `go vet ./...` then `go mod tidy` (run before committing)

Run one test: `go test ./router -run TestRoute_Auto`. Add `-tags onnx` when touching anything under `embed/` or `classifier/onnx_*`.

ONNX assets (`assets/`) are **gitignored** and regenerated via the Python script:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r tools/export-onnx/requirements.txt
python tools/export-onnx/export.py
```

The script also regenerates `testdata/classifier_golden.json` — the golden file the `-tags onnx` parity tests check against (numerics must match the Python reference within 1e-4).

## Repository layout — what's library vs glue

Public library packages (importable as `github.com/qiangli/nadir/...`) live at the **repo root**, not under `internal/`. This is intentional and was the point of commit `b91c982` ("Promote library packages out of internal/"). When adding code, decide first:

- **Library surface** (root packages: `classifier`, `router`, `embed`, `types`, `cache`, `ratelimit`, `health`, `modelmeta`, `provider/openai`, `provider/fake`) — reusable building blocks. Keep deps minimal. The top-level `nadir.go` re-exports the most common types so callers can write `nadir.Tier` / `nadir.RouteDecision` without importing sub-packages; keep that facade narrow.
- **App glue** (`internal/...`) — opinionated wiring that the binary needs but isn't reusable: HTTP server (`internal/proxy`), env-var loading (`internal/config`), SQLite + JSONL persistence (`internal/store`), Prometheus metrics, CLI reports, OAuth, setup wizard, dashboard.

`internal/config.Config.RouterConfig()` is the projection seam — the binary's superset config narrows down to the library-facing `router.Config` here. Add new library-relevant fields to both; binary-only options stay in `internal/config`.

## The routing pipeline (read once, then skim)

`router.Router.Route` mirrors `priorart/NadirClaw/nadirclaw/routing.py` step-for-step. The parity is enforced by `router/parity_test.go` against `testdata/routing_parity_corpus.json` — when changing routing logic, regenerate the corpus via `tools/parity-corpus/` rather than editing the JSON by hand.

Pipeline:
1. **Profile / alias / explicit-model.** `auto`/`smart`/empty → classify. Otherwise resolve via `modelAliases` (kept in sync with NadirClaw's `MODEL_ALIASES`).
2. **Modifiers.** Tools or agentic system-prompt → bump to complex. Reasoning markers (≥2 distinct matches of the `reasoningMarkers` regex — single-hit prompts intentionally do NOT trigger; the NadirClaw `set()` dedupe semantics are mirrored) → reasoning model. Images → vision-capable swap.
3. **Session pinning.** `SessionCache.UpgradeIfHigher` — once a conversation has been routed to a stronger model, later turns never drop back down.
4. **Fallback chain.** Configured chain + tier ladder (cheaper alternatives), dedup, drop primary, then `ProviderHealth.Order` reorders by recent failure rate.

`session` and `health` may be `nil`; the router skips those steps cleanly.

## The classifier tiers (composable)

Three stackable implementations behind `types.Classifier`:

1. **Heuristic** (`classifier.NewHeuristic`) — length + code-fence + keyword regex. Always available, zero deps. Used in tests and as the default build's only option.
2. **ONNX MiniLM** (`classifier.NewONNXFromAssets`, `-tags onnx` only) — pools embeddings against two pre-computed centroids. Numerics tracked against `testdata/classifier_golden.json`.
3. **Cascading** (`classifier.NewCascading`) — wraps any primary; when confidence is below threshold, asks a small LLM (typically Ollama llama3.2:3b) for a second opinion. **Silently falls back to the primary** on LLM timeout/error — routing must always produce an answer, never a 500.

The `cmd/nadir/classifier_select.go` (+ build-tag-split `classifier_primary_*.go`) wires these together. The **build tag is the strict-mode switch**: `-tags onnx` binaries fail startup if assets are missing — there is no silent heuristic fallback in production builds. Don't add an env-var override for this.

The classifier's chosen label (`"heuristic"` / `"onnx"` / `"cascade(onnx+llama3.2:3b)"`) is surfaced via `/health` so dashboards can alert when a deploy silently flips modes. Preserve that.

## Streaming vs batch fallback (proxy)

`internal/proxy` has two fallback semantics, and conflating them will break clients:

- **Batch**: cascade through `FallbackChain` transparently — client sees one final response or one error.
- **Streaming**: can only fall back **until the first SSE chunk goes out**. Once framing is committed, mid-stream errors surface as an SSE `error` event, not a fresh attempt on the next model.

`internal/proxy/fallback.go` and `stream.go` encode this — when modifying either, re-read the comment block at the top of `server.go`.

## NadirClaw parity

This is a clean-room derivative. Key invariants:

- Module path `github.com/qiangli/nadir`, binary `nadir`, env-var prefix `NADIR_*`, config dir `~/.nadir/`. The NadirClaw → nadir migration is a `sed` rename; semantics are preserved.
- The SQLite schema in `internal/store/sqlite` is **schema-compatible with NadirClaw's `request_logger.py`** — don't drift it without updating both sides of the migration doc.
- `priorart/NadirClaw/` is the upstream Python reference, checked in for parity comparison. Treat it as read-only.

## Conventions worth knowing

- Errors classified by `types.IsTransient` (re-exported as `nadir.IsTransientError`) drive the fallback retry loop. New provider errors should route through `types.ProviderError` so the loop sees them.
- `provider/openai` covers anything OpenAI-compatible: real OpenAI, Ollama, vLLM, LocalAI — distinguished by base URL. Don't add a new provider package for an OpenAI-shaped API; configure base URL instead.
- Don't widen `types.RequestMeta` for router-internal extraction needs; `router.meta` is the local bag for that.
- Logging is `log/slog` everywhere — no `fmt.Println` in library code.
