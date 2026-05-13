package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/internal/config"
	"github.com/qiangli/nadir/provider/openai"
	"github.com/qiangli/nadir/skillrouter"
)

// newSkillRouteCmd wires `nadir skill-route` — a one-shot CLI that
// picks a skill from a JSON catalog for a given prompt. Three modes:
//
//   - lexical  pure-Go TF-IDF, no Ollama (skillrouter.NewLexical)
//   - hybrid   TF-IDF primary + Ollama LLM rerank (skillrouter.NewHybrid)
//   - llm      LLM-only against the flat catalog (skillrouter.New)
//
// Default is "hybrid" when an LLM model is configured (via --model
// or NADIR_CASCADE_LLM_MODEL), otherwise "lexical". LLM config
// reuses the cascading-classifier env vars (NADIR_CASCADE_LLM_*) so
// the operator configures one small-model endpoint regardless of
// which feature consumes it.
func newSkillRouteCmd() *cobra.Command {
	var (
		skillsPath string
		prompt     string
		promptFile string
		mode       string
		model      string
		baseURL    string
		apiKey     string
		timeoutSec int
	)
	cmd := &cobra.Command{
		Use:   "skill-route",
		Short: "Pick the best skill / slash-command for a prompt",
		RunE: func(c *cobra.Command, _ []string) error {
			cfg := config.Load()

			// Resolve LLM connection params (used by hybrid + llm modes).
			if model == "" {
				model = cfg.CascadeLLMModel
			}
			if baseURL == "" {
				baseURL = cfg.CascadeLLMBaseURL
			}
			if baseURL == "" {
				baseURL = cfg.OllamaBaseURL
			}
			if apiKey == "" {
				apiKey = cfg.CascadeLLMAPIKey
			}
			if timeoutSec == 0 {
				timeoutSec = cfg.CascadeLLMTimeoutSec
			}

			// Resolve mode: explicit flag wins, then auto-detect from
			// whether an LLM model is available.
			if mode == "" {
				if model != "" {
					mode = "hybrid"
				} else {
					mode = "lexical"
				}
			}

			if skillsPath == "" {
				return errors.New("--skills <file> is required")
			}
			skills, err := loadSkillCatalog(skillsPath)
			if err != nil {
				return fmt.Errorf("load skills: %w", err)
			}

			text, err := resolvePromptInput(prompt, promptFile)
			if err != nil {
				return err
			}

			ctx, cancel := context.WithCancel(c.Context())
			defer cancel()

			matcher, label, err := buildSkillMatcher(ctx, mode, skills, baseURL, apiKey, model, timeoutSec)
			if err != nil {
				return err
			}

			d, err := matcher.Route(ctx, text)
			if err != nil {
				return err
			}

			out := map[string]any{
				"mode":         label,
				"skill":        d.Skill,
				"confidence":   d.Confidence,
				"margin":       d.Margin,
				"fell_through": d.FellThrough,
				"raw_response": d.RawResponse,
			}
			b, _ := json.MarshalIndent(out, "", "  ")
			fmt.Fprintln(os.Stdout, string(b))
			return nil
		},
	}
	cmd.Flags().StringVar(&skillsPath, "skills", "", "Path to JSON file with [{name, description, examples?}, ...]")
	cmd.Flags().StringVar(&prompt, "prompt", "", "Prompt text (use --prompt-file or stdin for longer input)")
	cmd.Flags().StringVar(&promptFile, "prompt-file", "", "Read prompt from this file (- for stdin)")
	cmd.Flags().StringVar(&mode, "mode", "", "Routing mode: lexical (TF-IDF, no LLM) | hybrid (TF-IDF + LLM rerank, default if LLM configured) | llm (LLM-only)")
	cmd.Flags().StringVar(&model, "model", "", "LLM model for hybrid/llm modes (defaults to NADIR_CASCADE_LLM_MODEL)")
	cmd.Flags().StringVar(&baseURL, "base-url", "", "LLM base URL (defaults to NADIR_CASCADE_LLM_BASE_URL, else NADIR_OLLAMA_BASE_URL)")
	cmd.Flags().StringVar(&apiKey, "api-key", "", "LLM API key (defaults to NADIR_CASCADE_LLM_API_KEY)")
	cmd.Flags().IntVar(&timeoutSec, "timeout-sec", 0, "Per-call LLM timeout for llm mode (defaults to NADIR_CASCADE_LLM_TIMEOUT_SEC)")
	return cmd
}

// buildSkillMatcher constructs the Matcher for the chosen mode and
// returns it along with a short label suitable for logging. lexical
// works without an LLM; hybrid and llm require one.
func buildSkillMatcher(ctx context.Context, mode string, skills []skillrouter.Skill, baseURL, apiKey, model string, timeoutSec int) (skillrouter.Matcher, string, error) {
	switch mode {
	case "lexical":
		m, err := skillrouter.NewLexical(skills)
		return m, "lexical", err
	case "hybrid":
		if model == "" {
			return nil, "", errors.New("hybrid mode needs --model or NADIR_CASCADE_LLM_MODEL")
		}
		client := openai.New("skill-router-llm", baseURL, apiKey)
		m, err := skillrouter.NewHybrid(ctx, client, model, skills)
		return m, "hybrid:" + model, err
	case "llm":
		if model == "" {
			return nil, "", errors.New("llm mode needs --model or NADIR_CASCADE_LLM_MODEL")
		}
		client := openai.New("skill-router-llm", baseURL, apiKey)
		opts := []skillrouter.Option{}
		if timeoutSec > 0 {
			opts = append(opts, skillrouter.WithTimeout(time.Duration(timeoutSec)*time.Second))
		}
		return skillrouter.New(client, model, skills, opts...), "llm:" + model, nil
	}
	return nil, "", fmt.Errorf("unknown --mode %q (lexical|hybrid|llm)", mode)
}

// loadSkillCatalog reads a JSON file shaped as
// [{"name": "...", "description": "...", "examples": ["...", ...]}, ...].
// Examples are optional but recommended for lexical/hybrid modes —
// they seed the TF-IDF prototypes that drive primary scoring.
func loadSkillCatalog(path string) ([]skillrouter.Skill, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw []struct {
		Name        string   `json:"name"`
		Description string   `json:"description"`
		Examples    []string `json:"examples"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		return nil, errors.New("catalog is empty")
	}
	out := make([]skillrouter.Skill, 0, len(raw))
	for _, r := range raw {
		if r.Name == "" {
			continue
		}
		out = append(out, skillrouter.Skill{
			Name:        r.Name,
			Description: r.Description,
			Examples:    r.Examples,
		})
	}
	if len(out) == 0 {
		return nil, errors.New("catalog has no entries with a non-empty name")
	}
	return out, nil
}

func resolvePromptInput(prompt, promptFile string) (string, error) {
	if prompt != "" && promptFile != "" {
		return "", errors.New("pass either --prompt or --prompt-file, not both")
	}
	if prompt != "" {
		return prompt, nil
	}
	if promptFile != "" {
		if promptFile == "-" {
			b, err := io.ReadAll(os.Stdin)
			return string(b), err
		}
		b, err := os.ReadFile(promptFile)
		return string(b), err
	}
	return "", errors.New("provide a prompt via --prompt or --prompt-file (use - for stdin)")
}
