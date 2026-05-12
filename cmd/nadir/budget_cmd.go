package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/internal/budget"
)

func newBudgetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "budget",
		Short: "Show the saved budget snapshot",
		RunE: func(_ *cobra.Command, _ []string) error {
			dir, err := nadirDataDir()
			if err != nil {
				return err
			}
			tr := budget.New(budget.Config{}, nil)
			if err := tr.LoadFrom(filepath.Join(dir, "budget_state.json")); err != nil {
				return err
			}
			b, _ := json.MarshalIndent(tr.Snapshot(), "", "  ")
			fmt.Fprintln(os.Stdout, string(b))
			return nil
		},
	}
}
