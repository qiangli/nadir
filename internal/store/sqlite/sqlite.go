// Package sqlite is the persistent request log. The schema is
// compatible with the Python implementation
// (priorart/NadirClaw/nadirclaw/request_logger.py) so a Python
// `nadir report` script could in principle read a Go-written DB.
package sqlite

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"github.com/qiangli/nadir/types"
)

type Store struct {
	db *sql.DB
}

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)")
	if err != nil {
		return nil, err
	}
	if err := initSchema(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

func (s *Store) Log(ctx context.Context, e *types.RequestEntry) error {
	if e.Timestamp.IsZero() {
		e.Timestamp = time.Now()
	}
	_, err := s.db.ExecContext(ctx,
		`INSERT INTO requests
			(id, ts, model, tier, provider, prompt_tokens, completion_tokens,
			 cost_usd, latency_ms, status, user_id, score, confidence, modifiers)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		e.ID, e.Timestamp.UTC().Format(time.RFC3339Nano), e.Model, string(e.Tier), e.Provider,
		e.PromptTokens, e.CompletionTokens, e.CostUSD, e.LatencyMs, e.Status, e.UserID,
		e.Score, e.Confidence, strings.Join(e.Modifiers, ","))
	return err
}

func (s *Store) Count(ctx context.Context) (int64, error) {
	var n int64
	err := s.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM requests").Scan(&n)
	return n, err
}

type QueryFilter struct {
	Since    time.Time
	Until    time.Time
	UserID   string
	Model    string
	Limit    int
}

func (s *Store) Query(ctx context.Context, f QueryFilter) ([]types.RequestEntry, error) {
	clauses := []string{}
	args := []any{}
	if !f.Since.IsZero() {
		clauses = append(clauses, "ts >= ?")
		args = append(args, f.Since.UTC().Format(time.RFC3339Nano))
	}
	if !f.Until.IsZero() {
		clauses = append(clauses, "ts < ?")
		args = append(args, f.Until.UTC().Format(time.RFC3339Nano))
	}
	if f.UserID != "" {
		clauses = append(clauses, "user_id = ?")
		args = append(args, f.UserID)
	}
	if f.Model != "" {
		clauses = append(clauses, "model = ?")
		args = append(args, f.Model)
	}
	where := ""
	if len(clauses) > 0 {
		where = " WHERE " + strings.Join(clauses, " AND ")
	}
	limit := 1000
	if f.Limit > 0 {
		limit = f.Limit
	}
	q := fmt.Sprintf(
		`SELECT id, ts, model, tier, provider, prompt_tokens, completion_tokens,
			cost_usd, latency_ms, status, user_id, score, confidence, modifiers
		 FROM requests%s ORDER BY ts DESC LIMIT %d`, where, limit)
	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []types.RequestEntry{}
	for rows.Next() {
		var e types.RequestEntry
		var ts, mods string
		var tier string
		if err := rows.Scan(&e.ID, &ts, &e.Model, &tier, &e.Provider,
			&e.PromptTokens, &e.CompletionTokens, &e.CostUSD, &e.LatencyMs,
			&e.Status, &e.UserID, &e.Score, &e.Confidence, &mods); err != nil {
			return nil, err
		}
		e.Tier = types.Tier(tier)
		if mods != "" {
			e.Modifiers = strings.Split(mods, ",")
		}
		if t, err := time.Parse(time.RFC3339Nano, ts); err == nil {
			e.Timestamp = t
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// Aggregate gives the report renderer a quick per-model + per-day
// breakdown without pulling every row.
type Aggregate struct {
	Model            string
	Day              string
	Count            int
	PromptTokens     int
	CompletionTokens int
	CostUSD          float64
	AvgLatencyMs     float64
}

func (s *Store) AggregateByModel(ctx context.Context, since time.Time) ([]Aggregate, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT model,
			COUNT(*),
			COALESCE(SUM(prompt_tokens),0),
			COALESCE(SUM(completion_tokens),0),
			COALESCE(SUM(cost_usd),0),
			COALESCE(AVG(latency_ms),0)
		 FROM requests WHERE ts >= ? GROUP BY model ORDER BY SUM(cost_usd) DESC`,
		since.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Aggregate{}
	for rows.Next() {
		var a Aggregate
		if err := rows.Scan(&a.Model, &a.Count, &a.PromptTokens, &a.CompletionTokens, &a.CostUSD, &a.AvgLatencyMs); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

func (s *Store) AggregateByDay(ctx context.Context, since time.Time) ([]Aggregate, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT substr(ts,1,10) AS day,
			COUNT(*),
			COALESCE(SUM(prompt_tokens),0),
			COALESCE(SUM(completion_tokens),0),
			COALESCE(SUM(cost_usd),0),
			COALESCE(AVG(latency_ms),0)
		 FROM requests WHERE ts >= ? GROUP BY day ORDER BY day DESC`,
		since.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Aggregate{}
	for rows.Next() {
		var a Aggregate
		if err := rows.Scan(&a.Day, &a.Count, &a.PromptTokens, &a.CompletionTokens, &a.CostUSD, &a.AvgLatencyMs); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

func initSchema(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS requests (
			id TEXT PRIMARY KEY,
			ts TEXT NOT NULL,
			model TEXT NOT NULL,
			tier TEXT NOT NULL,
			provider TEXT NOT NULL,
			prompt_tokens INTEGER NOT NULL DEFAULT 0,
			completion_tokens INTEGER NOT NULL DEFAULT 0,
			cost_usd REAL NOT NULL DEFAULT 0,
			latency_ms INTEGER NOT NULL DEFAULT 0,
			status TEXT NOT NULL DEFAULT '',
			user_id TEXT NOT NULL DEFAULT '',
			score REAL NOT NULL DEFAULT 0,
			confidence REAL NOT NULL DEFAULT 0,
			modifiers TEXT NOT NULL DEFAULT ''
		);
		CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
		CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
		CREATE INDEX IF NOT EXISTS idx_requests_user ON requests(user_id);
	`)
	return err
}

var _ types.RequestLogger = (*Store)(nil)
