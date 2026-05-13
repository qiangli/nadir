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
// asks the configured LLM to pick a skill from a JSON catalog for a
// given prompt. Useful for shell-pipeline integration and as a smoke
// test for the skillrouter package against a real Ollama instance.
//
// LLM config reuses the cascading-classifier env vars
// (NADIR_CASCADE_LLM_*) so the operator only configures one
// small-model endpoint regardless of which feature consumes it.
func newSkillRouteCmd() *cobra.Command {
	var (
		skillsPath string
		prompt     string
		promptFile string
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

			if model == "" {
				model = cfg.CascadeLLMModel
			}
			if model == "" {
				return errors.New("no LLM model: set NADIR_CASCADE_LLM_MODEL or pass --model")
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

			client := openai.New("skill-router-llm", baseURL, apiKey)
			opts := []skillrouter.Option{}
			if timeoutSec > 0 {
				opts = append(opts, skillrouter.WithTimeout(time.Duration(timeoutSec)*time.Second))
			}
			r := skillrouter.New(client, model, skills, opts...)

			ctx, cancel := context.WithCancel(c.Context())
			defer cancel()

			d, err := r.Route(ctx, text)
			if err != nil {
				return err
			}

			out := map[string]any{
				"skill":        d.Skill,
				"confidence":   d.Confidence,
				"fell_through": d.FellThrough,
				"raw_response": d.RawResponse,
			}
			b, _ := json.MarshalIndent(out, "", "  ")
			fmt.Fprintln(os.Stdout, string(b))
			return nil
		},
	}
	cmd.Flags().StringVar(&skillsPath, "skills", "", "Path to JSON file with [{name, description}, ...]")
	cmd.Flags().StringVar(&prompt, "prompt", "", "Prompt text (use --prompt-file or stdin for longer input)")
	cmd.Flags().StringVar(&promptFile, "prompt-file", "", "Read prompt from this file (- for stdin)")
	cmd.Flags().StringVar(&model, "model", "", "Override the LLM model (defaults to NADIR_CASCADE_LLM_MODEL)")
	cmd.Flags().StringVar(&baseURL, "base-url", "", "Override the LLM base URL (defaults to NADIR_CASCADE_LLM_BASE_URL, else NADIR_OLLAMA_BASE_URL)")
	cmd.Flags().StringVar(&apiKey, "api-key", "", "Override the LLM API key (defaults to NADIR_CASCADE_LLM_API_KEY)")
	cmd.Flags().IntVar(&timeoutSec, "timeout-sec", 0, "Per-call LLM timeout (defaults to NADIR_CASCADE_LLM_TIMEOUT_SEC, then 0=no extra timeout)")
	return cmd
}

// loadSkillCatalog reads a JSON file shaped as [{"name": "...", "description": "..."}, ...].
func loadSkillCatalog(path string) ([]skillrouter.Skill, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw []struct {
		Name        string `json:"name"`
		Description string `json:"description"`
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
		out = append(out, skillrouter.Skill{Name: r.Name, Description: r.Description})
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
