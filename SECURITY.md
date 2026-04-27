# Security policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive
findings. Instead, email the maintainer at
**ataknakbaba@gmail.com** with:

- A description of the issue.
- Steps to reproduce (a minimal repro is enough).
- The version / commit you tested against.
- Whether the issue lives in claimcheck itself or surfaces through one
  of the siblings.

We aim to acknowledge a report within 72 hours and to ship a fix in
the next minor release where applicable.

## Scope

In scope: the public Python API (`Pipeline`, `Verdict`), the example
scripts, the timing bench, and `from_daemon` / `from_corpus` /
`from_load` factories.

Out of scope:
- Issues that originate in `adaptmem` or `halluguard`. Report those
  upstream — links in the README. We will help triage.
- Bugs in third-party libraries (sentence-transformers, transformers,
  torch, fastapi, langchain). Report to the upstream project.

## Threat model — daemon mode

`Pipeline.from_daemon(daemon_url=...)` delegates encoding to a
long-lived `adaptmem serve` process. The daemon is **localhost-only,
single-user** by default. Do not point `daemon_url` at a public-
internet host without an auth proxy.
