package proxy

import (
	"context"
	"log/slog"
	"sync"

	"github.com/qiangli/nadir/internal/types"
)

// loggerPool fans out post-response logging across a bounded set of
// workers so a slow disk can't pile up unbounded goroutines. The pool
// is bounded by capacity, not workers — once the channel is full, new
// log entries are dropped (with a slog.Warn) rather than blocking the
// hot path.
type loggerPool struct {
	logger   *slog.Logger
	stores   []types.RequestLogger
	ch       chan *types.RequestEntry
	wg       sync.WaitGroup
	cancel   context.CancelFunc
	dropped  uint64
	closeOnce sync.Once
}

func newLoggerPool(logger *slog.Logger, stores []types.RequestLogger, workers, cap int) *loggerPool {
	if workers <= 0 {
		workers = 4
	}
	if cap <= 0 {
		cap = 1024
	}
	ctx, cancel := context.WithCancel(context.Background())
	p := &loggerPool{
		logger: logger,
		stores: stores,
		ch:     make(chan *types.RequestEntry, cap),
		cancel: cancel,
	}
	for range workers {
		p.wg.Add(1)
		go p.worker(ctx)
	}
	return p
}

func (p *loggerPool) Submit(e *types.RequestEntry) {
	select {
	case p.ch <- e:
	default:
		p.dropped++
		if p.logger != nil {
			p.logger.Warn("log entry dropped — pool saturated", slog.Uint64("dropped", p.dropped))
		}
	}
}

func (p *loggerPool) worker(ctx context.Context) {
	defer p.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case e, ok := <-p.ch:
			if !ok {
				return
			}
			for _, s := range p.stores {
				if err := s.Log(ctx, e); err != nil && p.logger != nil {
					p.logger.Warn("request log failed", slog.Any("err", err))
				}
			}
		}
	}
}

func (p *loggerPool) Close() {
	p.closeOnce.Do(func() {
		close(p.ch)
		p.cancel()
		p.wg.Wait()
	})
}
