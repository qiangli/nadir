# Backlog — Boss → Foreman → Worker

This is ycode's own task queue. The flow:

> A **Boss** (the human user) writes tasks as markdown files in
> `docs/backlog/`. A reconciler running inside `ycode serve` mirrors
> those files into Gitea issues in `admin/<project-slug>` every 60s.
> A **Foreman** (the agent the user is talking to — Claude Code, the
> ycode TUI, Cursor, Codex) picks the highest-priority unclaimed
> issue, leases a Loom workspace, and dispatches a **Worker** (a
> sandboxed `ycode autopilot worker` subprocess) to do the coding.
> The Worker opens a PR, the merger auto-merges on green CI, and the
> Foreman loops to the next task.

Source of truth: **`docs/backlog/<slug>.md` files**, not Gitea. Wipe
`~/.agents/ycode/gitea/` any time and the reconciler rebuilds the
queue from these markdown files.

## Source-of-truth contract

| | Authoritative for | Persistence | Recovery |
|---|---|---|---|
| `docs/backlog/<slug>.md` | task spec (title, body, priority, acceptance criteria) | git-tracked | survives Gitea wipe |
| Gitea issue `admin/<slug>#N` | runtime coordination state (claim labels, in-progress, comments) | `~/.agents/ycode/gitea/` (gitignored) | rebuilt by reconciler |
| Loom lease (`loom_id`) | worker workspace (clone + branch + author) | `~/.agents/ycode/gitea/loom/leases.json` | TTL-bounded; reaped after 8h |
| `.agents/ycode/foreman/{state,commands}` | Foreman lifecycle + Boss commands | gitignored | rebuilt on next start |

## Entry format

One `.md` file per task. YAML frontmatter, freeform markdown body.
Filename is `<slug>.md`; the slug is also the canonical identifier
(filename never changes — priority lives in frontmatter, not the
filename).

```markdown
---
title: Wire external_cnl executor through skill dispatcher
priority: p1
state: open
created: 2026-05-10T00:00:00Z
gitea_issue: 42
acceptance:
  - external_cnl skills resolve via dispatcher
  - smoke test green
---

Free-form markdown context here. Background, design notes, anything
the Worker should read before starting. The reconciler injects this
verbatim into the Gitea issue body, with a slug marker prefix for
re-linking after a Gitea wipe.
```

Required: `title`, `priority` (`p1|p2|p3`), `state` (`open|in_progress|done`).
Optional: `created`, `gitea_issue` (auto-populated by reconciler),
`acceptance` (rendered as `## Acceptance` in the Gitea body).

## Boss → Foreman → Worker chain of command

| Role | Holds | Receives instructions via |
|---|---|---|
| **Boss** (human) | everything | — |
| **Foreman** (the chat agent) | source tree, `docs/backlog/` read+write, `ycode backlog`/`ycode foreman` CLIs, full Gitea admin token, all MCP tools | in-band chat **or** out-of-band `ycode foreman <verb>` CLI; both write to `.agents/ycode/foreman/commands.jsonl` |
| **Worker** (`ycode autopilot worker`) | one Gitea issue, one Loom workspace, AST tools inside that workspace clone | SIGTERM from Foreman; comments/labels on its own Gitea issue |

The Worker never reads `docs/backlog/`, never invokes `ycode backlog`
or `ycode foreman`, and only knows about the issue it was handed and
the workspace path it was leased. v1 enforcement is conventional
(tool surface defined in spec, the Worker subprocess only registers
the two MCP servers it needs); per-Worker scoped Gitea tokens and
process-level sandboxing are deferred (see `docs/agent-collab.md:48`).

## Boss control protocol

The Boss steers the Foreman in real time via two equivalent surfaces.

**In-band (chat):** type instructions like "pause for now" or "skip
this and do dogfood-coverage next." The chat-agent Foreman translates
the intent into the equivalent verb and appends to
`.agents/ycode/foreman/commands.jsonl` for the audit trail, then
applies it.

**Out-of-band (CLI):**

```bash
ycode foreman start              # signal start (or resume)
ycode foreman pause              # finish current Worker, then idle
ycode foreman resume             # paused → running
ycode foreman stop               # graceful shutdown; SIGTERM mid-Worker
ycode foreman skip [--slug X]    # skip current (default) or named issue
ycode foreman prio <slug> p1|p2|p3   # re-rank a backlog entry
ycode foreman tell "<message>"   # freeform; Foreman LLM interprets
ycode foreman status             # print state.json
ycode foreman daemon             # headless state-machine watcher
```

State file at `.agents/ycode/foreman/state.json`:

```json
{
  "state": "running",
  "current_issue": 42,
  "current_loom_id": "loom-...",
  "current_slug": "cnl-executor",
  "started_at": "2026-05-10T11:00:00Z",
  "last_command_id": "01J...",
  "last_transition": "2026-05-10T11:30:00Z"
}
```

State machine: `idle` → (`start`) → `running` → (`pause`) → `paused` →
(`resume`) → `running` → (`stop`) → `stopped`. `start` from `paused`
is treated as `resume`.

## Kill-switch layers

Three layers, in order of granularity:

1. **`ycode foreman stop`** (Boss CLI / in-band chat) — graceful;
   cancels current Worker via SIGTERM, releases its claim, exits.
2. **`docs/backlog/PAUSE`** sentinel — file-based pause that survives
   Foreman restart and works without a running Foreman; checked
   between iterations. Touch the file to enter the next session in
   paused state; remove it to clear.
3. **Loom 8h `MaxTTL`** (`pkg/loom/types.go:120`) — process-level
   upper bound. Even if a Worker hangs, Loom will reap the lease.

## Reconciler

`ycode serve` runs an initial sync at startup, then polls every 60s.
Direction:

- **markdown → Gitea**: create missing issues, update drifted
  title/body/priority labels, close issues for `state: done`.
- **Gitea → markdown**: closed Gitea issues whose markdown is still
  `open|in_progress` get their `state:` flipped to `done`. This is how
  Worker completion (closes the issue) propagates back to the source
  of truth.

The reconciler is monotonic — it never deletes on either side. Orphan
Gitea issues from deleted markdown stay around. To remove an issue,
close it in Gitea (or set `state: done` in the markdown).

## Filesystem layout

```
ycode/
├── docs/backlog/
│   ├── README.md              (this file is at docs/backlog.md, not here)
│   ├── PAUSE                  (sentinel; kill-switch; gitignored)
│   ├── cnl-executor.md        (canonical task spec — committed)
│   ├── lighthouse-phase1.md
│   └── ...
├── .agents/ycode/foreman/     (gitignored)
│   ├── commands.jsonl         (Boss → Foreman command queue, append-only)
│   └── state.json             (Foreman → Boss state mirror)
└── ~/.agents/ycode/gitea/     (per-user, gitignored)
    ├── projects.json          (cwd → admin/<slug> mapping)
    ├── loom/leases.json       (active Loom workspace leases)
    └── repos/admin/<slug>/    (the Gitea repo data — wipe-safe)
```

## Build tag

All of the above is `//go:build experimental` (per `docs/strategy.md`).
Stable builds get no-op stubs. To use, build with:

```bash
go build -tags "sqlite,sqlite_unlock_notify,bindata,experimental" -o bin/ycode-exp ./cmd/ycode/
```

Graduation to `stable` requires meeting the criteria in
`docs/strategy.md` after dogfooding (one full week of Boss → Foreman →
Worker operation with telemetry collected).

## Related

- `docs/agent-collab.md` — the multi-agent runtime that the Worker
  inner loop builds on (collab orchestrator, merger, queue).
- `~/.config/ycode/skills/ycode-foreman/skill.md` — the Foreman loop
  body the chat agent follows when the Boss runs `/foreman`. User-global
  (universal across all repos); written by `ycode init`; embedded in
  the binary as the canonical fallback. Per-repo override: drop
  `.agents/ycode/skills/ycode-foreman/skill.md` in any project.
- `.agents/ycode/skills/ycode-autopilot/skill.md` — the inner
  RESEARCH→PLAN→BUILD→TEST→FIX→COMMIT loop the Worker runs in its
  Loom workspace.
- `internal/gitserver/backlog/` — reconciler implementation.
- `internal/gitserver/queue/` — priority queue + atomic claim/release.
- `pkg/loom/` — workspace lease lifecycle.
