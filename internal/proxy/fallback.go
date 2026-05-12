package proxy

import (
	"context"
	"errors"

	"github.com/qiangli/nadir/types"
)

// completeWithFallback runs the fallback loop for non-streaming
// requests. It tries each model in the chain in order; transient
// errors (rate-limit, 5xx, timeout, network) cascade to the next
// entry, fatal errors (auth, validation, bad-request) abort the chain
// and surface to the caller.
func (s *Server) completeWithFallback(ctx context.Context, req *types.ChatRequest, decision *types.RouteDecision, chain []string) (*types.ChatResponse, error) {
	var lastErr error
	for i, model := range chain {
		if !s.deps.Health.Available(model) {
			lastErr = &types.ProviderError{Kind: types.ErrRateLimit, Provider: "health", Model: model, Msg: "model in cooldown"}
			continue
		}
		if s.deps.ModelLimiter != nil {
			if _, ok := s.deps.ModelLimiter.Check(model); !ok {
				lastErr = &types.ProviderError{Kind: types.ErrRateLimit, Provider: "ratelimit", Model: model, Msg: "model in cooldown"}
				continue
			}
		}
		client, err := s.pickProvider(model)
		if err != nil {
			lastErr = err
			if !types.IsTransient(err) {
				return nil, err
			}
			continue
		}

		callCtx, cancel := context.WithTimeout(ctx, s.deps.PerCallTimeout)
		clone := *req
		clone.Model = model
		clone.Stream = false
		resp, callErr := client.Complete(callCtx, &clone)
		cancel()

		if callErr == nil {
			s.deps.Health.RecordSuccess(model)
			if i > 0 {
				// Track that we used a fallback — useful for metrics later.
				resp.Model = model
			}
			return resp, nil
		}

		s.recordFailure(model, callErr)
		lastErr = callErr
		if !types.IsTransient(callErr) {
			return nil, callErr
		}
	}
	if lastErr == nil {
		lastErr = errors.New("no models in fallback chain")
	}
	_ = decision
	return nil, lastErr
}

func (s *Server) recordFailure(model string, err error) {
	var pe *types.ProviderError
	kind := types.ErrUnknown
	if errors.As(err, &pe) {
		kind = pe.Kind
	}
	s.deps.Health.RecordFailure(model, kind)
	if s.deps.ModelLimiter != nil && kind == types.ErrRateLimit {
		s.deps.ModelLimiter.Record(model, 0)
	}
}
