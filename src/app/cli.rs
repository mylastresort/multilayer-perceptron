use std::error::Error;

use mlp::network::config::NetworkConfig;
use mlp::training::monitor::{MonitorMode, MonitoredMetric, parse_monitored_metrics};

use super::types::{CliArgs, MonitorOptions, NetOverrides, Subcommand};

pub fn default_dataset_path() -> String {
    format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"))
}

pub fn default_config_path() -> String {
    format!(
        "{}/models/training_learning_curve.yaml",
        env!("CARGO_MANIFEST_DIR")
    )
}

pub fn usage(binary_name: &str) -> String {
    format!(
        concat!(
            "Usage: {bin} <SUBCOMMAND> [OPTIONS]\n",
            "\n",
            "Subcommands:\n",
            "  split    Separate the dataset into train and validation CSV files\n",
            "  train    Train the network and save the model\n",
            "  predict  Load a saved model and evaluate on a dataset (binary cross-entropy)\n",
            "\n",
            "Common options:\n",
            "  -d, --dataset <PATH>               Dataset CSV path\n",
            "  -h, --help                         Print help\n",
            "\n",
            "split options:\n",
            "      --train-out <PATH>             Output CSV for training split\n",
            "      --val-out   <PATH>             Output CSV for validation split\n",
            "      --ratio     <FLOAT>            Training fraction (default 0.8)\n",
            "\n",
            "train options:\n",
            "  -c, --config    <PATH>             Network/training YAML config path\n",
            "      --model-out <PATH>             Where to save the trained model (JSON)\n",
            "  -v, --verbose                      Print loaded config summary\n",
            "  -g, --gui                          Open live learning-curve GUI\n",
            "  -M, --monitor-metrics <CSV>        Metrics to monitor/plot\n",
            "  -m, --monitor-primary <METRIC>     Metric for early stopping\n",
            "      --monitor-mode <MODE>          Early-stopping direction [min|max]\n",
            "  -p, --monitor-patience <INT>       Early-stopping patience in epochs\n",
            "      --monitor-min-delta <FLOAT>    Minimum improvement threshold\n",
            "  -s, --monitor-start-epoch <INT>    Epoch to start early-stopping checks\n",
            "      --monitor-early-stopping       Enable early stopping\n",
            "      --monitor-history-out <PATH>   Save per-epoch metric history to JSON\n",
            "  -l, --net-learning-rate <FLOAT>    Override config learning_rate\n",
            "  -e, --net-epochs <INT>             Override config epochs\n",
            "  -b, --net-batch-size <INT>         Override config batch_size\n",
            "\n",
            "predict options:\n",
            "      --model <PATH>                 Path to a saved model JSON\n",
            "\n",
            "Defaults:\n",
            "  --dataset {dataset}\n",
            "  --config  {config}",
        ),
        bin = binary_name,
        dataset = default_dataset_path(),
        config = default_config_path(),
    )
}

pub fn parse_cli_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut args = std::env::args();
    let binary_name = args.next().unwrap_or_else(|| "mlp".to_string());

    // Collect remaining args so we can peek at the subcommand first.
    let rest: Vec<String> = args.collect();

    // Handle top-level --help with no subcommand.
    if rest.is_empty() || rest[0] == "--help" || rest[0] == "-h" {
        println!("{}", usage(&binary_name));
        std::process::exit(0);
    }

    let subcommand = match rest[0].as_str() {
        "split" => Subcommand::Split,
        "train" => Subcommand::Train,
        "predict" => Subcommand::Predict,
        other => {
            return Err(
                format!("Unknown subcommand: '{other}'\n{}", usage(&binary_name)).into(),
            );
        }
    };

    let mut dataset_path = default_dataset_path();
    let mut config_path = default_config_path();
    let mut verbose = false;
    let mut gui = false;
    let mut monitor_options = MonitorOptions::default();
    let mut net_overrides = NetOverrides::default();
    let mut model_out: Option<String> = None;
    let mut model_in: Option<String> = None;
    let mut split_ratio: f64 = 0.8;
    let mut train_out: Option<String> = None;
    let mut val_out: Option<String> = None;

    let mut pending_flag: Option<String> = None;
    for arg in rest.into_iter().skip(1) {
        if let Some(flag) = pending_flag.take() {
            match flag.as_str() {
                "--dataset" | "-d" => dataset_path = arg,
                "--config" | "-c" => config_path = arg,
                "--model-out" => model_out = Some(arg),
                "--model" => model_in = Some(arg),
                "--train-out" => train_out = Some(arg),
                "--val-out" => val_out = Some(arg),
                "--ratio" => {
                    split_ratio = arg
                        .parse::<f64>()
                        .map_err(|_| format!("Invalid value for --ratio: {arg}"))?;
                }
                "--net-learning-rate" | "-l" => {
                    net_overrides.learning_rate =
                        Some(arg.parse::<f64>().map_err(|_| {
                            format!("Invalid value for --net-learning-rate: {arg}")
                        })?);
                }
                "--net-epochs" | "-e" => {
                    net_overrides.epochs = Some(
                        arg.parse::<usize>()
                            .map_err(|_| format!("Invalid value for --net-epochs: {arg}"))?,
                    );
                }
                "--net-batch-size" | "-b" => {
                    net_overrides.batch_size = Some(
                        arg.parse::<usize>()
                            .map_err(|_| format!("Invalid value for --net-batch-size: {arg}"))?,
                    );
                }
                "--monitor-metrics" | "-M" => {
                    monitor_options.metrics = parse_monitored_metrics(&arg)?;
                }
                "--monitor-primary" | "-m" => {
                    monitor_options.monitor_metric = MonitoredMetric::parse(&arg)
                        .ok_or_else(|| format!("Invalid value for --monitor-primary: {arg}"))?;
                }
                "--monitor-mode" => {
                    monitor_options.monitor_mode = MonitorMode::parse(&arg)
                        .ok_or_else(|| format!("Invalid value for --monitor-mode: {arg}"))?;
                }
                "--monitor-patience" | "-p" => {
                    monitor_options.monitor_patience = arg
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid value for --monitor-patience: {arg}"))?;
                }
                "--monitor-min-delta" => {
                    monitor_options.monitor_min_delta = arg
                        .parse::<f64>()
                        .map_err(|_| format!("Invalid value for --monitor-min-delta: {arg}"))?;
                }
                "--monitor-start-epoch" | "-s" => {
                    monitor_options.monitor_start_epoch = arg
                        .parse::<usize>()
                        .map_err(|_| format!("Invalid value for --monitor-start-epoch: {arg}"))?;
                }
                "--monitor-history-out" => monitor_options.history_out = Some(arg),
                _ => unreachable!("unsupported flag in parser state"),
            }
            continue;
        }

        match arg.as_str() {
            "--dataset"
            | "-d"
            | "--config"
            | "-c"
            | "--model-out"
            | "--model"
            | "--train-out"
            | "--val-out"
            | "--ratio"
            | "--net-learning-rate"
            | "-l"
            | "--net-epochs"
            | "-e"
            | "--net-batch-size"
            | "-b"
            | "--monitor-metrics"
            | "-M"
            | "--monitor-primary"
            | "-m"
            | "--monitor-mode"
            | "--monitor-patience"
            | "-p"
            | "--monitor-min-delta"
            | "--monitor-start-epoch"
            | "-s"
            | "--monitor-history-out" => pending_flag = Some(arg),
            "--verbose" | "-v" => verbose = true,
            "--gui" | "-g" => gui = true,
            "--monitor-early-stopping" => monitor_options.early_stopping = true,
            "--help" | "-h" => {
                println!("{}", usage(&binary_name));
                std::process::exit(0);
            }
            _ => {
                return Err(format!("Unknown argument: {arg}\n{}", usage(&binary_name)).into());
            }
        }
    }

    if let Some(flag) = pending_flag {
        return Err(format!("Missing value for {flag}\n{}", usage(&binary_name)).into());
    }

    Ok(CliArgs {
        subcommand,
        dataset_path,
        config_path,
        verbose,
        gui,
        monitor_options,
        net_overrides,
        model_out,
        model_in,
        split_ratio,
        train_out,
        val_out,
    })
}

pub fn apply_net_overrides(
    config: &mut NetworkConfig,
    overrides: &NetOverrides,
) -> Result<(), Box<dyn Error>> {
    if let Some(learning_rate) = overrides.learning_rate {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("--net-learning-rate must be a positive finite float".into());
        }
        config.learning_rate = learning_rate;
    }

    if let Some(epochs) = overrides.epochs {
        if epochs == 0 {
            return Err("--net-epochs must be greater than 0".into());
        }
        config.epochs = epochs;
    }

    if let Some(batch_size) = overrides.batch_size {
        if batch_size == 0 {
            return Err("--net-batch-size must be greater than 0".into());
        }
        config.batch_size = batch_size;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{apply_net_overrides, default_config_path, default_dataset_path, usage, NetOverrides};
    use mlp::network::config::NetworkConfig;

    fn make_config() -> NetworkConfig {
        let yaml = r#"
learning_rate: 0.01
epochs: 10
batch_size: 8
input_layers:
  - size: 4
hidden_layers:
  - size: 4
  - size: 4
output_layers:
  - size: 2
"#;
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn default_dataset_path_ends_with_data_csv() {
        let path = default_dataset_path();
        assert!(path.ends_with("data.csv"), "unexpected path: {path}");
    }

    #[test]
    fn default_config_path_ends_with_yaml() {
        let path = default_config_path();
        assert!(path.ends_with(".yaml"), "unexpected path: {path}");
    }

    #[test]
    fn usage_string_contains_all_subcommands() {
        let msg = usage("mlp");
        assert!(msg.contains("train"), "missing 'train'");
        assert!(msg.contains("split"), "missing 'split'");
        assert!(msg.contains("predict"), "missing 'predict'");
        assert!(msg.contains("Usage:"), "missing 'Usage:'");
    }

    #[test]
    fn apply_net_overrides_sets_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides { learning_rate: Some(0.001), ..NetOverrides::default() };
        apply_net_overrides(&mut config, &overrides).unwrap();
        assert!((config.learning_rate - 0.001).abs() < 1e-12);
    }

    #[test]
    fn apply_net_overrides_sets_epochs() {
        let mut config = make_config();
        let overrides = NetOverrides { epochs: Some(5), ..NetOverrides::default() };
        apply_net_overrides(&mut config, &overrides).unwrap();
        assert_eq!(config.epochs, 5);
    }

    #[test]
    fn apply_net_overrides_sets_batch_size() {
        let mut config = make_config();
        let overrides = NetOverrides { batch_size: Some(16), ..NetOverrides::default() };
        apply_net_overrides(&mut config, &overrides).unwrap();
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn apply_net_overrides_rejects_zero_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides { learning_rate: Some(0.0), ..NetOverrides::default() };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_negative_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides { learning_rate: Some(-0.01), ..NetOverrides::default() };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_infinite_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides { learning_rate: Some(f64::INFINITY), ..NetOverrides::default() };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_zero_epochs() {
        let mut config = make_config();
        let overrides = NetOverrides { epochs: Some(0), ..NetOverrides::default() };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_zero_batch_size() {
        let mut config = make_config();
        let overrides = NetOverrides { batch_size: Some(0), ..NetOverrides::default() };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_is_noop_when_all_none() {
        let mut config = make_config();
        apply_net_overrides(&mut config, &NetOverrides::default()).unwrap();
        assert!((config.learning_rate - 0.01).abs() < 1e-12);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 8);
    }
}
