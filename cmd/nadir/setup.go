package main

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"github.com/qiangli/nadir/internal/credentials"
	"github.com/qiangli/nadir/internal/setup"
)

func newSetupCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "setup",
		Short: "Interactive first-run wizard: writes credentials.json",
		RunE: func(cmd *cobra.Command, _ []string) error {
			dir, err := nadirDataDir()
			if err != nil {
				return err
			}
			w := setup.NewDefault(dir)
			return w.Run(cmd.Context())
		},
	}
}

func newAuthCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "auth <provider>",
		Short: "Store an API key for a provider (OAuth flows deferred to Phase 5)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			provider := strings.ToLower(args[0])
			token, _ := cmd.Flags().GetString("token")
			if token == "" {
				return fmt.Errorf("v1 requires --token flag; OAuth login is a Phase 5 feature")
			}
			dir, err := nadirDataDir()
			if err != nil {
				return err
			}
			s, err := credentials.Open(dir + "/credentials.json")
			if err != nil {
				return err
			}
			s.Set(provider, token)
			if err := s.Save(); err != nil {
				return err
			}
			fmt.Printf("Stored credentials for %s\n", provider)
			return nil
		},
	}
	cmd.Flags().String("token", "", "API key to store")
	return cmd
}
