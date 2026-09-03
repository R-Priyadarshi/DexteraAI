#!/usr/bin/env bash
# Run the test suite, and on failure republish the tail of the output as a
# GitHub annotation.
#
# Why this exists: downloading a job log needs admin rights on the repository,
# and annotations do not. For an ordinary failing assertion that gap does not
# matter, because pytest-github-actions-annotate-failures turns each failure
# into its own annotation. It matters when the process dies without pytest
# getting to report — a native library calling abort(), an OOM kill, a
# segfault. Then the plugin emits nothing, the log holds the only evidence, and
# anyone without admin rights is left guessing at a platform they cannot
# reproduce on. This puts the evidence somewhere they can read.
#
# Usage: scripts/run_tests_ci.sh [pytest args...]
set -uo pipefail

OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

pytest "$@" 2>&1 | tee "$OUT"
status=${PIPESTATUS[0]}

if [ "$status" -eq 0 ]; then
  exit 0
fi

# Name the signal when the shell reports one. A bare "exit code 134" is the
# thing that makes these crashes look inscrutable; 134 is 128 + 6 = SIGABRT,
# which says "a native library aborted" and rules out a Python-level failure
# before anyone starts reading.
signal=""
if [ "$status" -gt 128 ]; then
  case "$((status - 128))" in
    4)  signal=" (SIGILL — illegal instruction, often a wheel built for another CPU)" ;;
    6)  signal=" (SIGABRT — a native library called abort(), not a Python failure)" ;;
    9)  signal=" (SIGKILL — usually the runner running out of memory)" ;;
    11) signal=" (SIGSEGV — native crash)" ;;
    *)  signal=" (signal $((status - 128)))" ;;
  esac
fi

echo "pytest exited ${status}${signal}"

if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
  exit "$status"
fi

# One multi-line annotation rather than one per line: GitHub caps how many it
# will render, and a crash tail split across 40 annotations loses the ordering
# that makes it readable. Percent must be escaped before the newlines, or it
# would corrupt the %0A sequences written after it.
tail -n 40 "$OUT" \
  | sed -e 's/%/%25/g' -e 's/\r/%0D/g' \
  | awk 'BEGIN { ORS = "%0A" } { print }' \
  | { read -r -d '' body || true; \
      printf '::error title=pytest exited %s%s::%s\n' "$status" "$signal" "$body"; }

exit "$status"
