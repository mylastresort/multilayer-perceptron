# Branch Protection Setup

After creating your repo from this template, apply these rules manually in
**GitHub → Settings → Branches → Add rule → `main`**.

---

## Required Rules

| Rule | Value |
|------|-------|
| Require a pull request before merging | ✅ enabled |
| Required approvals | 1 (or more for teams) |
| Dismiss stale PR approvals on new commits | ✅ enabled |
| Require status checks to pass before merging | ✅ enabled |
| Require branches to be up to date | ✅ enabled |
| Block force pushes | ✅ enabled |
| Do not allow bypassing the above settings | ✅ enabled (applies to admins) |

---

## Required Status Checks

Add these checks — they map exactly to job names in the workflows:

| Check name | Workflow file |
|------------|---------------|
| `lint-commits` | `commitlint.yml` |
| `PR title follows conventional commits` | `pr-validation.yml` |
| `PR body has required sections` | `pr-validation.yml` |
| `perf PRs must update BENCHMARK.md` | `pr-validation.yml` |
| `Release & Changelog` | `release.yml` |

> Status check names must match exactly. Run each workflow at least once so
> GitHub discovers them before you add them as required checks.

---

## Secrets Required

| Secret | Used by | Required |
|--------|---------|----------|
| `GITHUB_TOKEN` | All workflows | Auto-provided by GitHub — never set manually |

---

## One-time Bootstrap (new repos only)

Release Please needs an existing tag to know where to start reading commits.
Run this **once** immediately after creating your repo from this template:

```bash
git tag v0.0.0
git push origin v0.0.0
```

Without this, the first release PR will not appear even with valid `feat:`
or `fix:` commits on main.

---

## How the Release Flow Works

```
feat: or fix: PR merged into main
          ↓
  Release Please workflow runs
          ↓
  Opens (or updates) a release PR
  titled "chore: release vX.Y.Z"
          ↓
  You review and merge the release PR
          ↓
  Release Please workflow runs again
  and detects it was its own PR
          ↓
  ┌──────────────────────────────────┐
  │ 1. Commits final CHANGELOG.md   │
  │ 2. Creates git tag vX.Y.Z       │
  │ 3. Creates GitHub Release        │
  │    with changelog as body        │
  └──────────────────────────────────┘
          ↓
  on-release job runs
  (publish / deploy / notify)
```

---

## Verifying a Release End-to-End

After the release PR merges, confirm all three happened:

| Where | What to check |
|-------|---------------|
| **Actions tab** | Release workflow ran green on the merge commit |
| **Releases page** | New release `vX.Y.Z` exists with full changelog body |
| **Tags page** | New tag `vX.Y.Z` points to the merge commit |

If the release and tag are missing but the workflow ran green, the most
likely cause is a missing `contents: write` permission in `release.yml`.

If no release PR appears after merging a `feat:` or `fix:` PR, check the
workflow run logs — Release Please prints either:

```
Created pull request #N: chore: release X.Y.Z
```

or:

```
No user facing changes to release
```

The second message means every commit since the last tag was `chore:`,
`docs:`, `ci:`, or `test:` — none of which qualify for a release.
