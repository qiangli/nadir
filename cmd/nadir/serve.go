package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/internal/budget"
	"github.com/qiangli/nadir/cache"
	"github.com/qiangli/nadir/classifier"
	"github.com/qiangli/nadir/internal/config"
	"github.com/qiangli/nadir/health"
	"github.com/qiangli/nadir/internal/metrics"
	"github.com/qiangli/nadir/modelmeta"
	"github.com/qiangli/nadir/provider/openai"
	"github.com/qiangli/nadir/internal/proxy"
	"github.com/qiangli/nadir/ratelimit"
	"github.com/qiangli/nadir/router"
	"github.com/qiangli/nadir/internal/store/jsonl"
	"github.com/qiangli/nadir/internal/store/sqlite"
	"github.com/qiangli/nadir/types"
)

func newServeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "serve",
		Short: "Start the nadir HTTP proxy",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runServe(cmd.Context())
		},
	}
}

func runServe(ctx context.Context) error {
	cfg := config.Load()

	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	thresh := classifier.Thresholds{
		Simple:  cfg.TierThresholds[0],
		Complex: cfg.TierThresholds[1],
		HasMid:  cfg.MidModel != "",
	}
	active, err := buildClassifier(logger, cfg, thresh)
	if err != nil {
		return err
	}
	cls := active.Classifier

	sess := cache.NewSession(30 * time.Minute)
	promptCache := cache.NewPrompt(cfg.CacheMaxSize, cfg.CacheTTL)
	tracker := health.New()
	userLimit := ratelimit.NewUser(cfg.UserRateWindow, cfg.UserRateLimit)
	modelLimit := ratelimit.NewModel()

	rt := router.New(cfg.RouterConfig(), cls, sess, tracker)

	providers, err := buildProviders(cfg)
	if err != nil {
		return err
	}

	dataDir, err := nadirDataDir()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return err
	}

	mt := modelmeta.Default()
	bud := budget.New(budget.Config{}, mt)
	_ = bud.LoadFrom(filepath.Join(dataDir, "budget_state.json"))

	mx := metrics.New()

	loggers := []types.RequestLogger{}
	sqliteStore, err := sqlite.Open(filepath.Join(dataDir, "requests.db"))
	if err != nil {
		logger.Warn("sqlite disabled", slog.Any("err", err))
	} else {
		loggers = append(loggers, sqliteStore)
	}
	jsonlStore, err := jsonl.Open(filepath.Join(dataDir, "requests.jsonl"))
	if err != nil {
		logger.Warn("jsonl disabled", slog.Any("err", err))
	} else {
		loggers = append(loggers, jsonlStore)
	}

	srv := proxy.New(proxy.Deps{
		Logger:       logger,
		Router:       rt,
		Classifier:   cls,
		Providers:    providers,
		PromptCache:  promptCache,
		UserLimiter:  userLimit,
		ModelLimiter: modelLimit,
		Health:       tracker,
		ModelToProvider: func(model string) string {
			if p, ok := cfg.ProviderForModel[model]; ok {
				return p
			}
			return "openai"
		},
		AuthToken:       cfg.AuthToken,
		PerCallTimeout:  2 * time.Minute,
		MaxBodyBytes:    int64(cfg.MaxBodyMB) * 1024 * 1024,
		Metrics:         mx,
		Loggers:         loggers,
		Budget:          bud,
		ClassifierLabel: active.Label,
	})
	defer srv.Close()

	httpSrv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 30 * time.Second,
	}

	// Warmup classifier so the first request doesn't pay init cost.
	go func() {
		if err := cls.Warmup(context.Background()); err != nil {
			logger.Warn("classifier warmup", slog.Any("err", err))
		}
	}()

	ctx, stop := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(shutdownCtx)
	}()

	logger.Info("nadir listening", slog.String("addr", cfg.Addr))
	if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func buildProviders(cfg *config.Config) (map[string]types.LLMClient, error) {
	out := map[string]types.LLMClient{}

	// OpenAI (or OpenAI-compatible base URL).
	if cfg.OpenAIAPIKey != "" || cfg.OpenAIBaseURL != "" {
		out["openai"] = openai.New("openai", cfg.OpenAIBaseURL, cfg.OpenAIAPIKey)
	}
	// Ollama is just an OpenAI-compatible client pointed at a local
	// base URL.
	if cfg.OllamaBaseURL != "" {
		out["ollama"] = openai.New("ollama", cfg.OllamaBaseURL, "")
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("no providers configured: set NADIR_OPENAI_API_KEY, NADIR_OPENAI_BASE_URL, or NADIR_OLLAMA_BASE_URL")
	}
	return out, nil
}
