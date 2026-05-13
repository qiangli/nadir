// Command nadir is the entry point for the proxy binary. Phase 1
// only wires the `serve` subcommand. Phase 3+ add `report`, `budget`,
// `cache`, `savings`, `optimize`, `models`, `setup`, `dashboard`,
// `auth`.
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

const version = "0.1.0-phase1"

func main() {
	root := &cobra.Command{
		Use:     "nadir",
		Short:   "OpenAI-compatible LLM proxy with embedding-based routing",
		Version: version,
	}
	root.AddCommand(
		newServeCmd(),
		newReportCmd(),
		newSavingsCmd(),
		newBudgetCmd(),
		newModelsCmd(),
		newSetupCmd(),
		newAuthCmd(),
		newDashboardCmd(),
		newSkillRouteCmd(),
	)

	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
