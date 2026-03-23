## Summary

<!-- What changed and why. 2–3 sentences max. -->

## Type of change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `perf` — performance improvement
- [ ] `refactor` — no behavior change
- [ ] `chore` / `ci` / `docs`

## How to test

```bash
# Steps to verify this PR works
```

---

## ✅ Proven Results

<!--
Fill this section ONLY if you have hard numbers from benchmarks, logs, or profiling.
perf PRs: this section is REQUIRED and must contain at least one number.
All other types: fill if relevant, leave empty otherwise.
-->

- **Before:** <!-- e.g. p99 latency 320ms at 200 req/s -->
- **After:** <!-- e.g. p99 latency 48ms at 200 req/s -->
- **Delta:** <!-- e.g. 85% reduction -->
- **Measured with:** <!-- e.g. k6, wrk, pytest-benchmark, pprof -->
- **Conditions:** <!-- e.g. 200 concurrent users, 60s sustained load, m5.large -->

---

## Checklist

- [ ] Commit messages follow [conventional commits](../CONTRIBUTING.md)
- [ ] Tests added or updated
- [ ] `perf` PR: `BENCHMARK.md` updated with new measurements
- [ ] `perf` PR: "Proven Results" section filled with numbers
