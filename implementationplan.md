# Implementation plan — AI-SETI

Author: Inventions4All — github:TWeb79

Pending work only. Completed tasks are removed once done. The cleanup of 2026-10-03 removed
tasks 1–35 of the 0.3.0 hardening pass and every backlog item from B1 to B45: all fixed, except
B10 and B40, which were closed with measured reasons (see "Verified — not bugs" in
[backlog.md](backlog.md)). Their write-ups, test plans and verification notes are in git
history.

## Task list

No pending tasks. New work starts as an item in [backlog.md](backlog.md).

## Test debt

`cli.py` and `dashboard.py` have the least unit coverage (see backlog "Test debt"). Don't raise
the 60% CI floor until one of them has real tests.
