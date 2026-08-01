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

## Prerequisites

The live GUI monitor (`-g`) and the crate's plotting dependencies pull in
`fontconfig` and `freetype` through `yeslogic-fontconfig-sys`. Its build
script calls `pkg-config` to find `fontconfig.pc` and **panics if it cannot
locate it**, so install the native dev packages *before* building:

- **Debian / Ubuntu**
  ```bash
  sudo apt install libfontconfig1-dev libfreetype6-dev libbzip2-dev
  ```
- **Fedora / RHEL**
  ```bash
  sudo dnf install fontconfig-devel freetype-devel bzip2-devel
  ```
- **Arch Linux**
  ```bash
  sudo pacman -S fontconfig freetype2
  ```
- **macOS** (Homebrew)
  ```bash
  brew install fontconfig freetype
  ```

This repository ships a `.cargo/config.toml` that sets `PKG_CONFIG_PATH` to
the Debian/Ubuntu multiarch pkg-config directories, so Ubuntu-style setups
work out of the box (the value is harmless elsewhere). On Linux the GUI also
requires an X11 display. If you still hit the `fontconfig.pc ... not found`
panic on a machine where the packages *are* installed, your `pkg-config` is
not searching that platform's prefix — add the right directory to
`PKG_CONFIG_PATH` in `.cargo/config.toml`.

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
	-M loss,accuracy,precision
```

Enable early stopping (on by default; restores the best-epoch weights):

```bash
cargo run -- train --model-out models/model.json
```

Early stopping is enabled by default with a patience of 60, so the saved model
is rolled back to the best epoch (`restore_best_weights`). Tune it explicitly:

```bash
cargo run -- train \
	--monitor-early-stopping \
	-m loss \
	--monitor-mode min \
	-p 5 \
	--monitor-min-delta 0.0001 \
	-s 2
```

Disable it entirely (saves the final-epoch weights):

```bash
cargo run -- train --no-early-stopping --model-out models/model.json
```

Save training history JSON:

```bash
cargo run -- \
	--monitor-history-out reports/history.json
```

### GUI Learning Curves

Open live GUI panels for monitored metrics:

```bash
cargo run -- -g -M loss,accuracy,precision
```

GUI controls:

- Press `Space` to close the live window.

Optional GUI sizing/refresh controls:

```bash
MLP_LIVE_PLOT_WIDTH=1600 \
MLP_LIVE_PLOT_HEIGHT=1000 \
MLP_LIVE_PLOT_DELAY_MS=10 \
cargo run -- -g -M loss,accuracy,precision
```

## Documentation

- [Benchmarks & Performance](./BENCHMARK.md)
- [Changelog](./CHANGELOG.md)
- [Contributing](./CONTRIBUTING.md)

## License

This project is released under the license specified in this repository.
