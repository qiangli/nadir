---
id: b502a943c702
kind: bug
title: S557 make log rotation preserve writer ownership on Windows
seq: 1
status: assigned
priority: p1
created: 2026-11-01T17:08:24.29663Z
assignee: s557-manager
sprint: 557
sprint_id: f5ccf563-f5d1-5b94-8be4-9388e51ef86d
sprint_title: 'Bashy v1.0.0 preparation: split private/public development and retire qiangli/dhnt'
---

Native Windows logmaint rename fails because writer remains open; gzip cleanup also removes input before closing. Implement serialized close/rename/reopen transaction with recovery on rename failure and checked closes before removal. Preserve rows and rotation behavior, test partition and failure recovery, verify native remote.
