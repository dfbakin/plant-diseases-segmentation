"""Regression tests for ``scripts/run_seg_probes_overnight.sh``.

Background
----------
Two real bugs slipped into the overnight orchestrator and went unnoticed
until the 18 Apr run silently marched all three phases as ✓ when in fact
Phase 1 had crashed mid-eval:

1. **Broken exit-code capture in ``run_phase``**::

       if ! wait "$CHILD_PID"; then
           ec=$?    # always 0 -- the '!' has already negated the pipeline
       fi

   ``$?`` inside an ``if ! cmd; then ...; fi`` body is the result of the
   *negated* pipeline, not the original command. Empirically::

       bash -c 'if ! (exit 7); then echo $?; fi'   # -> 0  (BUG)
       bash -c '(exit 7); echo $?'                 # -> 7  (correct)

2. **Chain didn't short-circuit on failure**::

       run_phase "phase1" ...
       run_phase "phase2" ...   # ran even if phase1 failed

   Combined with bug 1 this meant every phase script's exit code was
   discarded, every phase got reported "✓ complete", and the master
   summary cheerfully announced "ALL PHASES COMPLETE in 34 min" while
   only 2 of 11 Phase-1 probes actually had results.

These tests assert both bugs stay fixed:

* :class:`TestOrchestratorScript` lints the file -- catches regressions
  the moment someone re-types the broken pattern.
* :class:`TestBashExitCapturePattern` proves at runtime that the
  *replacement* pattern propagates exit codes correctly across the
  same bash version we ship with.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ORCHESTRATOR = Path("scripts/run_seg_probes_overnight.sh")


@pytest.fixture(scope="module")
def script_text() -> str:
    assert ORCHESTRATOR.is_file(), f"missing {ORCHESTRATOR}"
    return ORCHESTRATOR.read_text()


class TestOrchestratorScript:
    """Static guards on the orchestrator's source -- catch regressions."""

    def test_no_broken_if_not_wait_pattern(self, script_text: str) -> None:
        """The ``if ! wait ...; then ec=$?; fi`` anti-pattern must not return.

        ``$?`` in the body is the negated pipeline result (always 0), so
        every failed phase used to be silently reported as ✓.
        """
        # Match `if ! wait` on a single line followed by `ec=$?` within
        # the next few lines (allowing for whitespace and the body).
        broken = re.compile(
            r"if\s*!\s*wait\b[^\n]*\n\s*ec=\$\?",
            re.MULTILINE,
        )
        assert broken.search(script_text) is None, (
            "Found re-introduced `if ! wait $CHILD_PID; then ec=$?; fi` "
            "pattern -- $? inside the body is the *negated* exit (always 0). "
            "Use `wait $CHILD_PID || ec=$?` instead."
        )

    def test_uses_correct_exit_capture_pattern(self, script_text: str) -> None:
        """The fix must be present: ``wait "$CHILD_PID" || ec=$?``."""
        assert 'wait "$CHILD_PID" || ec=$?' in script_text, (
            "Expected `wait \"$CHILD_PID\" || ec=$?` -- this is the only "
            "pattern that correctly propagates the awaited process's exit."
        )

    def test_each_run_phase_call_has_short_circuit(self, script_text: str) -> None:
        """Every ``run_phase`` invocation must be followed by ``|| { ...; exit ...; }``.

        Without the short-circuit the chain runs all phases even when an
        earlier one died -- that's how Phase 2 + 3 silently fired with
        missing inputs (``ERROR: ... selected.json missing``) and still
        got reported as ✓ complete.
        """
        invocations = re.findall(
            r'run_phase\s+"phase\d+"[^\n]*',
            script_text,
        )
        assert len(invocations) >= 3, (
            f"Expected 3 run_phase invocations, found {len(invocations)}; "
            "the orchestrator structure may have changed."
        )

        # Accepted forms (single-line is what we ship, but be liberal about
        # whitespace so future reformatting doesn't break the test):
        #   run_phase "phaseN" ... || {
        #   run_phase "phaseN" ... || exit ...
        # Both must appear on the *same* logical line as the run_phase
        # invocation -- a bare ``run_phase "phaseN" ...`` with no ``||``
        # is the bug.
        run_phase_with_guard = re.compile(
            r'run_phase\s+"phase\d+"[^\n]*\|\|\s*(?:\{|exit\b)',
        )
        guarded = run_phase_with_guard.findall(script_text)
        assert len(guarded) == len(invocations), (
            f"Found {len(invocations)} run_phase invocations but only "
            f"{len(guarded)} have a `|| {{ ... }}` / `|| exit ...` short-circuit. "
            "Every run_phase call must guard the chain against silent "
            "downstream-phase invocation on failure."
        )

        # Spot-check the guard runs `exit` (not just `log`), so a failed
        # phase actually terminates the orchestrator.
        for inv in invocations:
            assert re.search(r"\|\|\s*(?:\{|exit\b)", inv), inv
            # If using the block form, the body (next ~5 lines) must
            # contain `exit`. Find the matching block in the full text.
            if "|| {" in inv:
                block_start = script_text.find(inv)
                block_end = script_text.find("}", block_start)
                assert block_end != -1, f"Unterminated block after: {inv}"
                body = script_text[block_start:block_end]
                assert "exit" in body, (
                    f"Guard block must contain `exit` to short-circuit "
                    f"the chain. Found:\n{body}"
                )


class TestBashExitCapturePattern:
    """Runtime proof that our chosen pattern correctly propagates exit codes.

    These mirror the empirical experiments that uncovered the bug. They
    intentionally use ``/bin/bash`` (not ``sh``) since the orchestrator's
    shebang is ``#!/usr/bin/env bash`` and the ``$?``-after-``!`` quirk
    is bash-specific.
    """

    @staticmethod
    def _run(snippet: str) -> str:
        """Run a tiny bash snippet and return its stdout (stripped)."""
        result = subprocess.run(
            ["bash", "-c", snippet],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()

    def test_broken_pattern_loses_exit_code(self) -> None:
        """Documents the bug we fixed: the ``if !`` form silently zeros $?."""
        out = self._run('if ! (exit 7); then echo "$?"; fi')
        # If this ever changes, bash semantics changed and our reasoning
        # may need revisiting. Today (bash 5.x) it returns 0.
        assert out == "0", (
            f"Expected the broken pattern to return 0 (the bug), got {out!r}. "
            "If bash semantics have changed, the orchestrator's defensive "
            "comment may be misleading."
        )

    def test_fixed_pattern_preserves_exit_code(self) -> None:
        """``cmd || ec=$?`` correctly captures the child's real exit code."""
        out = self._run('ec=0; (exit 7) || ec=$?; echo "$ec"')
        assert out == "7", f"Expected 7, got {out!r} -- the fix is broken."

    def test_fixed_pattern_zero_on_success(self) -> None:
        """``cmd || ec=$?`` leaves ``ec`` at its default (0) when cmd succeeds."""
        out = self._run('ec=0; true || ec=$?; echo "$ec"')
        assert out == "0"

    def test_fixed_pattern_with_wait(self) -> None:
        """End-to-end: backgrounded child + ``wait || ec=$?`` propagates 7."""
        snippet = (
            'set -uo pipefail; '
            '(sleep 0.05; exit 7) & '
            'CHILD_PID=$!; '
            'ec=0; '
            'wait "$CHILD_PID" || ec=$?; '
            'echo "$ec"'
        )
        out = self._run(snippet)
        assert out == "7", (
            f"Expected wait to propagate child exit 7, got {out!r}. "
            "This is the exact production code path."
        )
