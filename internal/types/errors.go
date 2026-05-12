package types

import (
	"errors"
	"fmt"
)

// ErrorKind classifies upstream and internal errors for the fallback
// loop. Transient kinds (RateLimit, ServerError, Timeout, Network) tell
// the proxy to try the next model in the chain; Fatal kinds (Auth,
// Validation, BadRequest) abort the chain and surface to the caller.
type ErrorKind int

const (
	ErrUnknown ErrorKind = iota
	ErrRateLimit
	ErrServerError
	ErrTimeout
	ErrNetwork
	ErrAuth
	ErrValidation
	ErrBadRequest
	ErrCanceled
)

func (k ErrorKind) String() string {
	switch k {
	case ErrRateLimit:
		return "rate_limit"
	case ErrServerError:
		return "server_error"
	case ErrTimeout:
		return "timeout"
	case ErrNetwork:
		return "network"
	case ErrAuth:
		return "auth"
	case ErrValidation:
		return "validation"
	case ErrBadRequest:
		return "bad_request"
	case ErrCanceled:
		return "canceled"
	default:
		return "unknown"
	}
}

// ProviderError is what every LLMClient must return when the upstream
// rejects a call. Wrapping the kind explicitly lets the fallback loop
// branch on transient-vs-fatal without parsing error strings.
type ProviderError struct {
	Kind       ErrorKind
	StatusCode int
	Provider   string
	Model      string
	Msg        string
	Underlying error
}

func (e *ProviderError) Error() string {
	if e.Underlying != nil {
		return fmt.Sprintf("%s (%s/%s status=%d): %s: %v", e.Kind, e.Provider, e.Model, e.StatusCode, e.Msg, e.Underlying)
	}
	return fmt.Sprintf("%s (%s/%s status=%d): %s", e.Kind, e.Provider, e.Model, e.StatusCode, e.Msg)
}

func (e *ProviderError) Unwrap() error { return e.Underlying }

// IsTransient reports whether the fallback loop should keep trying the
// next model in the chain. Auth and validation errors are fatal.
func IsTransient(err error) bool {
	var pe *ProviderError
	if errors.As(err, &pe) {
		switch pe.Kind {
		case ErrRateLimit, ErrServerError, ErrTimeout, ErrNetwork:
			return true
		}
	}
	return false
}
