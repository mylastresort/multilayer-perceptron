# Multilayer Perceptron (Rust)

Train and evaluate a configurable MLP from CSV data with YAML-based network/training config,
live monitoring, early stopping, and metric history export.

[![Codecov](https://codecov.io/gh/mylastresort/multilayer-perceptron/graph/badge.svg?token=jmjvtSTUeJ)](https://codecov.io/gh/mylastresort/multilayer-perceptron)

## Getting Started

```bash
git clone https://github.com/mylastresort/multilayer-perceptron
cd multilayer-perceptron
cargo run
```

## Usage

Show all CLI options:

```bash
cargo run -- --help
```

Run with defaults (`data/data.csv` + `models/training_learning_curve.yaml`):

```bash
cargo run
```

Run with explicit dataset and config paths:

```bash
cargo run -- \
	--dataset /absolute/path/to/data.csv \
	--config /absolute/path/to/config.yaml
```

Print resolved config summary:

```bash
cargo run -- --verbose
```

Override config values from CLI:

```bash
cargo run -- -l 0.02 -e 50 -b 16
```

### Monitoring

Monitor multiple metrics each epoch:

```bash
cargo run -- \
	-M loss,accuracy,precision,recall,f1
```

Enable early stopping:

```bash
cargo run -- \
	--monitor-early-stopping \
	-m loss \
	--monitor-mode min \
	-p 5 \
	--monitor-min-delta 0.0001 \
	-s 2
```

Save training history JSON:

```bash
cargo run -- \
	--monitor-history-out reports/history.json
```

### GUI Learning Curves

Open live GUI panels for monitored metrics:

```bash
cargo run -- -g -M loss,accuracy,f1
```

GUI controls:

- Press `Space` to close the live window.

Optional GUI sizing/refresh controls:

```bash
MLP_LIVE_PLOT_WIDTH=1600 \
MLP_LIVE_PLOT_HEIGHT=1000 \
MLP_LIVE_PLOT_DELAY_MS=10 \
cargo run -- -g -M loss,accuracy,f1
```

## Documentation

- [Benchmarks & Performance](./BENCHMARK.md)
- [Changelog](./CHANGELOG.md)
- [Contributing](./CONTRIBUTING.md)

## License

This project is released under the license specified in this repository.
