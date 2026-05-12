# docs/backlog/

Canonical task list. **One `.md` per task, slug = filename stem.**
See [`docs/backlog.md`](../backlog.md) for the source-of-truth
contract, the Boss → Foreman → Worker chain, the Boss control
protocol, and the reconciler semantics.

This `README.md` is not an issue — the reconciler skips it.

## Adding a new task

```bash
ycode backlog new "Implement <feature>" --priority p1
ycode backlog list                  # show all
ycode backlog list --priority p1    # only top tier
ycode backlog show <slug>           # render one
```

The reconciler (running inside `ycode serve`) syncs new entries to
Gitea on its next 60s poll; force a sync with `ycode backlog reconcile`.
