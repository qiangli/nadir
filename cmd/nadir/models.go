package main

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/modelmeta"
)

func newModelsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "models",
		Short: "List known models with cost and context window",
		RunE: func(_ *cobra.Command, _ []string) error {
			tw := tabwriter.NewWriter(os.Stdout, 0, 2, 2, ' ', 0)
			fmt.Fprintln(tw, "model\tinput/1k\toutput/1k\tcontext\tvision")
			for _, m := range modelmeta.Default().All() {
				vis := "no"
				if m.Vision {
					vis = "yes"
				}
				fmt.Fprintf(tw, "%s\t%g\t%g\t%d\t%s\n", m.Name, m.InputPer1K, m.OutputPer1K, m.Context, vis)
			}
			return tw.Flush()
		},
	}
}
