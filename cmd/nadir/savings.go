package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/modelmeta"
	"github.com/qiangli/nadir/internal/savings"
	"github.com/qiangli/nadir/internal/store/sqlite"
)

func newSavingsCmd() *cobra.Command {
	var since string
	var baseline string
	cmd := &cobra.Command{
		Use:   "savings",
		Short: "Compute cost saved vs an always-premium baseline",
		RunE: func(_ *cobra.Command, _ []string) error {
			dur, err := time.ParseDuration(since)
			if err != nil {
				return fmt.Errorf("--since: %w", err)
			}
			dir, err := nadirDataDir()
			if err != nil {
				return err
			}
			store, err := sqlite.Open(filepath.Join(dir, "requests.db"))
			if err != nil {
				return err
			}
			defer store.Close()
			calc := savings.New(store, modelmeta.Default())
			rep, err := calc.Compute(context.Background(), time.Now().Add(-dur), baseline)
			if err != nil {
				return err
			}
			b, _ := json.MarshalIndent(rep, "", "  ")
			fmt.Fprintln(os.Stdout, string(b))
			return nil
		},
	}
	cmd.Flags().StringVar(&since, "since", "168h", "lookback window")
	cmd.Flags().StringVar(&baseline, "baseline", "", "baseline model (default: most expensive in modelmeta)")
	return cmd
}
