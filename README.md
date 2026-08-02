# Multilayer Perceptron (Rust)

Train and evaluate a configurable MLP from CSV data with YAML-based network/training config,
per-epoch metrics, static learning-curve PNGs, early stopping, and metric history export.

[![Codecov](https://codecov.io/gh/mylastresort/multilayer-perceptron/graph/badge.svg?token=jmjvtSTUeJ)](https://codecov.io/gh/mylastresort/multilayer-perceptron)

## Getting Started

```bash
git clone https://github.com/mylastresort/multilayer-perceptron
cd multilayer-perceptron
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	--model-out models/model.json
```

## Prerequisites

The crate's plotting dependencies (used to render the learning-curve PNGs) pull in
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
work out of the box (the value is harmless elsewhere). If you still hit the
`fontconfig.pc ... not found` panic on a machine where the packages *are*
installed, your `pkg-config` is not searching that platform's prefix — add the
right directory to `PKG_CONFIG_PATH` in `.cargo/config.toml`.

## Usage

Show all CLI options:

```bash
cargo run -- --help
```

Train with defaults (`data/data.csv` + `models/mandatory_sgd.yaml`):

```bash
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	--model-out models/model.json
```

`--dataset` and `--config` are required for `train` (no implicit default);
`split` and `predict` fall back to `data/data.csv` when `--dataset` is omitted.

Train with explicit dataset and config paths:

```bash
cargo run -- train \
	--dataset /absolute/path/to/data.csv \
	--config /absolute/path/to/config.yaml
```

Print resolved config summary:

```bash
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	--verbose
```

Override config values from CLI:

```bash
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	-l 0.02 -e 50 -b 16
```

### Monitoring

Per-epoch training and validation metrics are printed to the console for every
epoch, and after training a single learning-curve PNG is written to `reports/`.
The filename embeds the YAML config name and a timestamp so each trained model
gets its own image (e.g. `reports/learning_curves_mandatory_sgd_20260802-104713.png`),
and the chart carries all curves of that model on one graph:

- training and validation loss per epoch
- training and validation accuracy per epoch
- training and validation precision per epoch

Enable early stopping (on by default; restores the best-epoch weights):

```bash
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	--model-out models/model.json
```

Early stopping is enabled by default with a patience of 60, so the saved model
is rolled back to the best epoch (`restore_best_weights`). Tune it explicitly:

```bash
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	--early-stopping \
	--early-stop-metric loss \
	--early-stop-mode min \
	--early-stop-patience 5 \
	--early-stop-min-delta 0.0001 \
	--early-stop-start-epoch 2
```

Disable it entirely (saves the final-epoch weights):

```bash
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	--no-early-stopping \
	--model-out models/model.json
```

Save training history JSON:

```bash
cargo run -- train \
	--dataset data/data.csv \
	--config models/mandatory_sgd.yaml \
	--monitor-history-out reports/history.json
```

## Scripts

`scripts/train_model.sh` runs the full pipeline interactively: it prompts for a
dataset (default `data/data.csv`) and a numbered config menu of every shipped
model, then splits, trains and evaluates in one shot.

Each shipped config also has its own dedicated script, so a single command
trains and evaluates with that config and its bonus flags baked in:

| Script | Config | Demonstrates |
| --- | --- | --- |
| `scripts/train_mandatory_sgd.sh` | `models/mandatory_sgd.yaml` | Mandatory SGD + categorical cross-entropy |
| `scripts/train_bonus_adam.sh` | `models/bonus_adam.yaml` | Bonus: Adam optimizer + weight decay |
| `scripts/train_bonus_early_stopping.sh` | `models/bonus_early_stopping.yaml` | Bonus: early stopping with best-epoch restore |
| `scripts/train_bonus_history.sh` | `models/bonus_history.yaml` | Bonus: per-epoch history JSON + multi-metric curves |
| `scripts/train_bonus_multiple_curves.sh` | `models/bonus_multiple_curves.yaml` | Bonus: monitors loss/accuracy/precision, multiple curves on one graph |
| `scripts/train_bonus_precision.sh` | `models/bonus_precision.yaml` | Bonus: precision as the early-stopping metric |

Override the dataset and model path via the `DATASET` and `MODEL_OUT`
environment variables:

```bash
DATASET=data/data_test.csv MODEL_OUT=models/model.json ./scripts/train_bonus_multiple_curves.sh
```

## Documentation

- [Benchmarks & Performance](./BENCHMARK.md)
- [Changelog](./CHANGELOG.md)
- [Contributing](./CONTRIBUTING.md)

## License

This project is released under the license specified in this repository.
