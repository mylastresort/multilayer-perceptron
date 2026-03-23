# Contributing

## Commit Convention

This repo enforces [Conventional Commits](https://www.conventionalcommits.org/).
Every commit message must follow this format:

```
<type>(<scope>): <description>
```

### Allowed types

| Type       | When to use                                      |
|------------|--------------------------------------------------|
| `feat`     | New feature or capability                        |
| `fix`      | Bug fix                                          |
| `perf`     | Performance improvement — **must include numbers** |
| `refactor` | Code change with no behavior change              |
| `chore`    | Tooling, deps, config                            |
| `docs`     | Documentation only                               |
| `ci`       | Workflow / pipeline changes                      |
| `test`     | Adding or fixing tests                           |

### Examples

```bash
# ✅ Good
feat(auth): add OAuth2 PKCE flow for mobile clients
fix(api): prevent race condition under high concurrency
perf(parser): reduce p99 latency 320ms → 48ms via state machine
refactor(db): replace N+1 queries with batch fetch

# ❌ Rejected by commitlint
fix: bug fix
perf(api): improve performance   # no numbers = rejected
updated stuff
```

> `perf` commits **must** contain a measurable metric in the description.
> No number = commitlint blocks the PR.

---

## Pull Requests

1. Branch from `main`
2. Follow the PR template — fill every section relevant to your change
3. `perf` PRs must also update `BENCHMARK.md` with new measurements
4. All status checks must pass before merge is allowed

## Branch Protection

Direct pushes to `main` are blocked. All changes go through a PR.
Force pushes are disabled. Use `git revert` to undo a merged commit.
