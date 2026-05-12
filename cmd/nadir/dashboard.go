package main

import (
	"fmt"
	"net/http"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/internal/budget"
	"github.com/qiangli/nadir/internal/dashboard"
	"github.com/qiangli/nadir/internal/store/sqlite"
	"github.com/qiangli/nadir/modelmeta"
)

func newDashboardCmd() *cobra.Command {
	var addr string
	cmd := &cobra.Command{
		Use:   "dashboard",
		Short: "Serve a read-only HTML dashboard on a separate port",
		RunE: func(cmd *cobra.Command, _ []string) error {
			dir, err := nadirDataDir()
			if err != nil {
				return err
			}
			store, err := sqlite.Open(filepath.Join(dir, "requests.db"))
			if err != nil {
				return err
			}
			defer store.Close()
			tr := budget.New(budget.Config{}, modelmeta.Default())
			_ = tr.LoadFrom(filepath.Join(dir, "budget_state.json"))

			mux := http.NewServeMux()
			mux.Handle("/", dashboard.Handler(dashboard.Deps{Store: store, Budget: tr, Table: modelmeta.Default()}))

			fmt.Fprintf(cmd.OutOrStdout(), "nadir dashboard listening on http://%s\n", addr)
			srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
			return srv.ListenAndServe()
		},
	}
	cmd.Flags().StringVar(&addr, "addr", ":8857", "listen address")
	return cmd
}
