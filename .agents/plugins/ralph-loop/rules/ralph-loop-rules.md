# Ralph Loop Operational Rules

1. **Integrity Guard**: Never comment out, delete, or weaken test cases, assertions, or validation checks just to make the test runner exit with code 0.
2. **Bounded Iteration**: Always cap the execution loop to a maximum iteration limit (default 5) to prevent infinite cycles.
3. **Traceback Fidelity**: Always inspect the complete stack trace before writing a patch; do not guess error locations.
4. **Minimal Patch Principle**: Fix only what is broken to satisfy the test contract; avoid unprompted large-scale refactoring during a fix cycle.
5. **Clear Audit Trail**: Document each iteration's hypothesis and applied change in the final summary.
