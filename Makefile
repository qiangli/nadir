.DEFAULT_GOAL := help

BIN     := nadir
BINDIR  := bin
GOFLAGS := -trimpath
LDFLAGS := -s -w

.PHONY: help tidy build build-onnx test test-onnx clean

help: ## Show this help (default target)
	@awk 'BEGIN {FS = ":.*?## "; printf "\nUsage: make <target>\n\nTargets:\n"} \
		/^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

tidy: ## Run go mod tidy
	go mod tidy

build: ## Build the default binary (heuristic classifier, no ML deps)
	@mkdir -p $(BINDIR)
	go build $(GOFLAGS) -ldflags "$(LDFLAGS)" -o $(BINDIR)/$(BIN) ./cmd/nadir
	@echo "built $(BINDIR)/$(BIN)"

build-onnx: ## Build with MiniLM ONNX classifier (requires CGO + libonnxruntime)
	@mkdir -p $(BINDIR)
	go build $(GOFLAGS) -tags onnx -ldflags "$(LDFLAGS)" -o $(BINDIR)/$(BIN) ./cmd/nadir
	@echo "built $(BINDIR)/$(BIN) (onnx)"

test: ## Run unit tests on the default build
	go test ./...

test-onnx: ## Run unit tests including the ONNX classifier path
	go test -tags onnx ./...

clean: ## Remove build artifacts
	rm -rf $(BINDIR)
	go clean -testcache
