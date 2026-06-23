# Contributing

`whichvlm` is a Python 3.11+ CLI for ranking local VLM options against real
or simulated hardware. Contributions should be small, tested, and easy to
review.

## Setup

From a fresh clone:

```bash
git clone https://github.com/ptxv/whichvlm.git
cd whichvlm
uv sync --group dev
uv run pytest -q
```

For focused work, run the tests that cover the touched code:

```bash
uv run pytest -q tests/test_ranker.py
uv run python -m compileall -q src tests
```

## Issues

Include the exact command, output or traceback, OS, GPU or CPU, and whether the
run used real or simulated hardware. Ranking issues should also include the
expected top result and the actual top result.

## Code

- Match the existing module style before adding new structure.
- Keep the patch to the behavior under review; do not mix in cleanup.
- Use concrete names. Avoid generic helpers such as `process_data`,
  `handle_response`, or `validate_input`.
- Avoid leading-underscore names for new helpers or locals.
- Do not add one-use dataclasses, config objects, logging wrappers, retry
  wrappers, or CLI boilerplate.
- Let bad inputs fail loudly unless the caller already has a concrete recovery
  path.
- Write comments only for intent that is not obvious from the code.
- Add tests for changed behavior, boundary cases, and failure modes.

## Agent Use

Agent-assisted PRs are welcome when the author owns the diff. Before opening a
PR, read every changed line and remove invented APIs, broad abstractions,
dead code, filler comments, and tests that only mirror the implementation.

Do not claim hardware, performance, or runtime support without evidence from
the device, command output, benchmark, log, or failing test that backs it up.

## Pull Requests

Use a short title that names the change, such as `Fix AMD shared-memory
detection`.

PR descriptions should be brief and specific:

- What changed and why
- The main files or modules touched
- Exact tests run and observed result
- Known gaps, especially hardware or runtime paths not tested

For docs-only changes, say that no code tests were needed. For code changes,
prefer focused tests first; run the full suite when touching shared ranking,
fetching, runtime, output, or hardware-detection behavior.
