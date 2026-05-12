// Package setup is the interactive first-run wizard: it prompts for
// provider API keys, optionally scans for Ollama instances, and
// writes ~/.nadir/credentials.json + .env.
//
// The wizard intentionally accepts blank input for any field — there
// is no provider that is mandatory, since the user might run nadir as
// an Ollama-only proxy.
package setup

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/qiangli/nadir/internal/credentials"
	"github.com/qiangli/nadir/internal/ollama"
)

type Wizard struct {
	In       io.Reader
	Out      io.Writer
	DataDir  string
	Discover func(ctx context.Context) []ollama.Instance
}

func NewDefault(dataDir string) *Wizard {
	disc := ollama.NewDefault()
	return &Wizard{
		In:      os.Stdin,
		Out:     os.Stdout,
		DataDir: dataDir,
		Discover: func(ctx context.Context) []ollama.Instance {
			return disc.Discover(ctx)
		},
	}
}

func (w *Wizard) Run(ctx context.Context) error {
	if err := os.MkdirAll(w.DataDir, 0o755); err != nil {
		return err
	}
	fmt.Fprintln(w.Out, "nadir setup")
	fmt.Fprintln(w.Out, "===========")
	fmt.Fprintln(w.Out, "Provide API keys for any providers you want to use.")
	fmt.Fprintln(w.Out, "Leave blank to skip a provider. Press enter to continue.")
	fmt.Fprintln(w.Out)

	r := bufio.NewReader(w.In)
	creds, err := credentials.Open(filepath.Join(w.DataDir, "credentials.json"))
	if err != nil {
		return err
	}

	prompts := []struct {
		provider string
		label    string
	}{
		{"openai", "OpenAI API key (sk-…)"},
		{"anthropic", "Anthropic API key (sk-ant-…)"},
		{"google", "Google AI Studio key (AIza…)"},
	}
	for _, p := range prompts {
		fmt.Fprintf(w.Out, "%s: ", p.label)
		line, err := r.ReadString('\n')
		if err != nil && err != io.EOF {
			return err
		}
		v := strings.TrimSpace(line)
		if v != "" {
			creds.Set(p.provider, v)
		}
	}

	// Ollama discovery.
	fmt.Fprintln(w.Out)
	fmt.Fprintln(w.Out, "Scanning for local Ollama instances…")
	scanCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	instances := w.Discover(scanCtx)
	cancel()
	if len(instances) == 0 {
		fmt.Fprintln(w.Out, "  (none found — that's fine if you don't run Ollama)")
	} else {
		for _, inst := range instances {
			fmt.Fprintf(w.Out, "  found %s with %d models\n", inst.BaseURL, len(inst.Models))
		}
	}

	if err := creds.Save(); err != nil {
		return err
	}
	fmt.Fprintln(w.Out)
	fmt.Fprintf(w.Out, "Saved credentials to %s\n", filepath.Join(w.DataDir, "credentials.json"))
	fmt.Fprintln(w.Out, "Start the proxy: nadir serve")
	return nil
}
