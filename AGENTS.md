# StereoForge Repository Rules

## Mandatory Pre-Commit Code Simplification & Repository Hygiene Rule

1. **Mandatory Simplification & Cleanup Iteration Before Every Commit:**
   - Every commit **MUST** pass an dedicated code simplification and cleanup iteration before being committed.
   - During this iteration, review all touched and related files to:
     - Simplify control flow, eliminate redundant helpers/branches, and tighten data structures.
     - Remove dead code, commented-out blocks, unused imports, and duplicate logic.
     - Remove stale code, obsolete one-off scripts, superseded documentation/evidence clutter, and stale untracked artifacts so the repository remains clean and tidy.

2. **Verification & Git Discipline:**
   - Every commit must pass relevant CTest and `pytest` suites (including `-m gpu` when GPU/shader/benchmark paths are modified) after the simplification and cleanup pass.
   - Always push verified commits to `origin/main`.
