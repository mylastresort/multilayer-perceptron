# Benchmarks & Performance

> **Last updated:** <!-- DATE -->  
> **Version:** <!-- e.g. v1.4.2 -->  
> **Benchmark runner:** <!-- e.g. k6 / wrk / hyperfine / pytest-benchmark / custom -->

This document captures performance measurements across all project dimensions: system-level metrics (latency, throughput, memory, CPU) and project-specific metrics that reflect what this software actually does at scale.

---

## Table of Contents

1. [Test Environment](#test-environment)
2. [How to Run Benchmarks](#how-to-run-benchmarks)
3. [Latency & Response Time](#latency--response-time)
4. [Throughput](#throughput)
5. [Memory Usage](#memory-usage)
6. [CPU & Resource Consumption](#cpu--resource-consumption)
7. [Project-Specific Performance](#project-specific-performance)
8. [Version History](#version-history)
9. [Interpreting Results](#interpreting-results)

---

## Test Environment

All benchmarks were collected under the following conditions. Results from different environments are **not** directly comparable.

| Property          | Value                                      |
|-------------------|--------------------------------------------|
| OS                | <!-- e.g. Ubuntu 22.04 LTS -->             |
| CPU               | <!-- e.g. AMD Ryzen 9 7950X, 16-core -->   |
| RAM               | <!-- e.g. 64 GB DDR5-5200 -->              |
| Storage           | <!-- e.g. NVMe SSD, 3500 MB/s read -->     |
| Network           | <!-- e.g. Loopback / 1 Gbps LAN -->        |
| Runtime / SDK     | <!-- e.g. Node.js 20.11, Go 1.22 -->       |
| Compiler flags    | <!-- e.g. -O2, --release -->               |
| Warm-up           | <!-- e.g. 10 s warm-up before recording -->|
| Iterations        | <!-- e.g. 3 runs, median reported -->       |
| Date collected    | <!-- YYYY-MM-DD -->                        |

> ⚠️ Cloud VM results may show higher variance. Pin CPU frequency and disable turbo-boost for reproducible results.

---

## How to Run Benchmarks

### Prerequisites

```bash
# Install benchmark dependencies
<!-- e.g. npm install / pip install -r requirements-bench.txt / go install ... -->
```

### Full suite

```bash
<!-- e.g. make bench / npm run benchmark / cargo bench -->
```

### Individual suites

```bash
# Latency only
<!-- command -->

# Throughput only
<!-- command -->

# Memory profile
<!-- command -->

# Project-specific
<!-- command -->
```

### Reproducing a specific result

```bash
# Example: reproduce the p99 latency result for /api/process
<!-- e.g. k6 run --vus 50 --duration 60s bench/latency.js -->
```

---

## Latency & Response Time

> Measured at **steady state** with the service fully warmed up.  
> All times in **milliseconds (ms)** unless noted.

### Summary

| Scenario                  | p50    | p95    | p99    | p999   | Max    |
|---------------------------|--------|--------|--------|--------|--------|
| <!-- Scenario A -->       | <!-- -->| <!-- -->| <!-- -->| <!-- -->| <!-- -->|
| <!-- Scenario B -->       | <!-- -->| <!-- -->| <!-- -->| <!-- -->| <!-- -->|
| <!-- Scenario C -->       | <!-- -->| <!-- -->| <!-- -->| <!-- -->| <!-- -->|

### Latency Distribution — `<!-- primary scenario -->`

```
  p50  ██████████████████████████  XX ms
  p75  ████████████████████████████████  XX ms
  p90  ████████████████████████████████████  XX ms
  p95  ██████████████████████████████████████████  XX ms
  p99  ██████████████████████████████████████████████████  XX ms
 p999  ████████████████████████████████████████████████████████  XX ms
```

### Notes

- <!-- e.g. "p99 spike at 320 ms caused by GC pause; see issue #42" -->
- <!-- e.g. "Cold-start adds ~180 ms on first request after idle" -->

---

## Throughput

> Measured in **requests per second (req/s)** or **operations per second (ops/s)**.  
> Concurrent users / workers are listed per row.

### HTTP / API Throughput

| Concurrency | Req/s  | Avg Latency | Error Rate |
|-------------|--------|-------------|------------|
| 1           | <!-- -->| <!-- -->    | <!-- -->   |
| 10          | <!-- -->| <!-- -->    | <!-- -->   |
| 50          | <!-- -->| <!-- -->    | <!-- -->   |
| 100         | <!-- -->| <!-- -->    | <!-- -->   |
| 500         | <!-- -->| <!-- -->    | <!-- -->   |

### Saturation Point

> The system saturates (error rate > 1%) at approximately **<!-- N --> concurrent users / <!-- N --> req/s**.

### Queue / Worker Throughput *(if applicable)*

| Queue depth | Jobs/s | Worker count | Avg job duration |
|-------------|--------|--------------|------------------|
| 100         | <!-- -->| <!-- -->    | <!-- -->         |
| 1 000       | <!-- -->| <!-- -->    | <!-- -->         |
| 10 000      | <!-- -->| <!-- -->    | <!-- -->         |

### CLI / Library Throughput *(if applicable)*

| Input size  | Ops/s  | Notes                  |
|-------------|--------|------------------------|
| 1 KB        | <!-- -->| <!-- -->              |
| 1 MB        | <!-- -->| <!-- -->              |
| 100 MB      | <!-- -->| <!-- -->              |

---

## Memory Usage

> Measured after **steady-state** operation (post-warm-up, no GC pressure).  
> All values in **MiB** unless noted.

### Baseline & Under Load

| Scenario             | RSS    | Heap / Alloc | Peak    | After GC |
|----------------------|--------|--------------|---------|----------|
| Idle / startup       | <!-- -->| <!-- -->    | <!-- --> | <!-- --> |
| 50 concurrent users  | <!-- -->| <!-- -->    | <!-- --> | <!-- --> |
| 100 concurrent users | <!-- -->| <!-- -->    | <!-- --> | <!-- --> |
| Max observed load    | <!-- -->| <!-- -->    | <!-- --> | <!-- --> |

### Memory Growth Over Time

> Run the following to check for leaks:

```bash
<!-- e.g. node --expose-gc bench/memleak.js -->
```

| Time elapsed | RSS    | Notes              |
|--------------|--------|--------------------|
| 0 min        | <!-- -->| Baseline          |
| 10 min       | <!-- -->| <!-- -->          |
| 30 min       | <!-- -->| <!-- -->          |
| 60 min       | <!-- -->| Stable / growing? |

### Notes

- <!-- e.g. "Heap grows ~2 MB per 10k processed items due to caching; eviction kicks in at 512 MB" -->

---

## CPU & Resource Consumption

> Measured with `<!-- e.g. perf, pprof, py-spy, async-profiler -->`.

### CPU Usage

| Scenario             | Avg CPU % | Peak CPU % | Cores used |
|----------------------|-----------|------------|------------|
| Idle                 | <!-- -->  | <!-- -->   | <!-- -->   |
| 50 req/s             | <!-- -->  | <!-- -->   | <!-- -->   |
| 200 req/s            | <!-- -->  | <!-- -->   | <!-- -->   |
| Saturation           | <!-- -->  | <!-- -->   | <!-- -->   |

### Hot Paths (Top Functions by CPU time)

| Rank | Function / Symbol         | % CPU  | Notes              |
|------|---------------------------|--------|--------------------|
| 1    | <!-- -->                  | <!-- -->| <!-- -->          |
| 2    | <!-- -->                  | <!-- -->| <!-- -->          |
| 3    | <!-- -->                  | <!-- -->| <!-- -->          |

> Full flame graph: `<!-- link or path to flamegraph.svg -->`

### I/O & Syscalls *(if applicable)*

| Metric               | Value   |
|----------------------|---------|
| Disk read (avg)      | <!-- --> |
| Disk write (avg)     | <!-- --> |
| Network tx (avg)     | <!-- --> |
| Network rx (avg)     | <!-- --> |
| Syscalls/req         | <!-- --> |

---

## Project-Specific Performance

> These metrics measure **what this project uniquely does** — not generic system performance.  
> Replace the examples below with metrics that reflect your domain.

---

### Example: Data Pipeline

| Stage              | Rows/s  | Latency p50 | Notes                   |
|--------------------|---------|-------------|-------------------------|
| Ingestion          | <!-- -->| <!-- -->    | <!-- -->                |
| Transformation     | <!-- -->| <!-- -->    | <!-- -->                |
| Aggregation        | <!-- -->| <!-- -->    | <!-- -->                |
| Output / export    | <!-- -->| <!-- -->    | <!-- -->                |

---

### Example: ML / Inference

| Model              | Input size | Batch size | Inference time | Throughput  |
|--------------------|------------|------------|----------------|-------------|
| <!-- Model A -->   | <!-- -->   | 1          | <!-- --> ms    | <!-- --> /s |
| <!-- Model A -->   | <!-- -->   | 32         | <!-- --> ms    | <!-- --> /s |
| <!-- Model B -->   | <!-- -->   | 1          | <!-- --> ms    | <!-- --> /s |

---

### Example: CLI / Build Tool

| Task               | Input         | Time (real) | Time (user) | Time (sys) |
|--------------------|---------------|-------------|-------------|------------|
| <!-- e.g. parse -->| <!-- size --> | <!-- -->    | <!-- -->    | <!-- -->   |
| <!-- e.g. build -->| <!-- size --> | <!-- -->    | <!-- -->    | <!-- -->   |
| Incremental build  | <!-- delta -->| <!-- -->    | <!-- -->    | <!-- -->   |

---

### Example: Frontend / UI

| Metric                        | Value    | Target  |
|-------------------------------|----------|---------|
| Lighthouse Performance score  | <!-- --> | ≥ 90    |
| First Contentful Paint (FCP)  | <!-- --> | < 1.8 s |
| Largest Contentful Paint (LCP)| <!-- --> | < 2.5 s |
| Total Blocking Time (TBT)     | <!-- --> | < 200 ms|
| Cumulative Layout Shift (CLS) | <!-- --> | < 0.1   |
| Bundle size (gzip)            | <!-- --> | < 200 KB|
| Time to Interactive (TTI)     | <!-- --> | < 3.8 s |

---

### Example: Database / Storage Engine

| Operation    | Dataset size | Rows/s or MB/s | p99 latency |
|--------------|--------------|----------------|-------------|
| INSERT       | 1 M rows     | <!-- -->       | <!-- -->    |
| SELECT (idx) | 1 M rows     | <!-- -->       | <!-- -->    |
| SELECT (seq) | 1 M rows     | <!-- -->       | <!-- -->    |
| UPDATE       | 1 M rows     | <!-- -->       | <!-- -->    |
| Bulk load    | 10 GB        | <!-- -->       | <!-- -->    |

---

## Version History

Track regressions and improvements across releases.

| Version   | Date       | Throughput (req/s) | p99 Latency | Memory (MB) | Notes                        |
|-----------|------------|--------------------|-------------|-------------|------------------------------|
| <!-- v --> | <!-- --> | <!-- -->           | <!-- -->    | <!-- -->    | <!-- e.g. baseline -->       |
| <!-- v --> | <!-- --> | <!-- -->           | <!-- -->    | <!-- -->    | <!-- e.g. added cache -->    |
| <!-- v --> | <!-- --> | <!-- -->           | <!-- -->    | <!-- -->    | <!-- e.g. regression fixed -->|

---

## Interpreting Results

### What the percentiles mean

| Percentile | Meaning                                                     |
|------------|-------------------------------------------------------------|
| p50        | Median — typical user experience                            |
| p95        | 95% of requests are faster than this                        |
| p99        | Worst 1% — affects power users and high-traffic periods     |
| p999       | Worst 0.1% — rare spikes, often GC or I/O jitter           |

### Known performance trade-offs

- <!-- e.g. "Caching improves p99 by 4× but adds 40 MB baseline memory overhead" -->
- <!-- e.g. "Enabling compression halves bandwidth at the cost of +12% CPU" -->
- <!-- e.g. "Batch processing 10× faster than item-by-item but adds 2–5 s latency per batch" -->

### What is NOT benchmarked here

- <!-- e.g. "Multi-region / geo-distributed latency" -->
- <!-- e.g. "Performance under disk saturation" -->
- <!-- e.g. "Cold container start times in Kubernetes" -->

### Contributing benchmark results

1. Run the full suite on your hardware (see [How to Run](#how-to-run-benchmarks)).
2. Open a PR with your results in a new column or separate environment table.
3. Include full environment specs.

---

*Generated with care. Keep this file updated with each release.*
<!-- test -->
