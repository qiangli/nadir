package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/internal/report"
	"github.com/qiangli/nadir/internal/store/sqlite"
)

func newReportCmd() *cobra.Command {
	var since string
	var byModel, byDay, asHTML bool
	cmd := &cobra.Command{
		Use:   "report",
		Short: "Render a usage report from the request log",
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
			r := report.New(store)
			return r.Render(context.Background(), os.Stdout, report.Options{
				Since: dur, ByModel: byModel, ByDay: byDay, HTML: asHTML,
			})
		},
	}
	cmd.Flags().StringVar(&since, "since", "24h", "lookback window (e.g. 24h, 7d)")
	cmd.Flags().BoolVar(&byModel, "by-model", true, "per-model breakdown")
	cmd.Flags().BoolVar(&byDay, "by-day", false, "per-day breakdown")
	cmd.Flags().BoolVar(&asHTML, "html", false, "render as HTML")
	return cmd
}
