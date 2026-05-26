use std::error::Error;

use mlp::network::config::NetworkConfig;
use mlp::training::monitor::{MonitorMode, MonitoredMetric, parse_monitored_metrics};

use super::types::{CliArgs, MonitorOptions, NetOverrides};

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
            "Usage: {} [OPTIONS]\n",
            "\n",
            "Options:\n",
            "  -d, --dataset <PATH>               Dataset CSV path\n",
            "  -c, --config <PATH>                Network/training YAML config path\n",
            "  -v, --verbose                      Print loaded config in a visual summary\n",
            "  -g, --gui                          Open live learning-curve GUI window\n",
            "  -h, --help                         Print help\n",
            "\n",
            "Monitoring:\n",
            "  -M, --monitor-metrics <CSV>        Metrics to monitor/plot\n",
            "                                      Allowed: loss,accuracy,precision,recall,f1\n",
            "  -m, --monitor-primary <METRIC>     Metric used for early stopping\n",
            "      --monitor-mode <MODE>          Early-stopping direction [min|max]\n",
            "  -p, --monitor-patience <INT>       Early-stopping patience in epochs\n",
            "      --monitor-min-delta <FLOAT>    Minimum improvement threshold\n",
            "  -s, --monitor-start-epoch <INT>    Epoch to start early-stopping checks\n",
            "      --monitor-early-stopping       Enable early stopping\n",
            "      --monitor-history-out <PATH>   Save per-epoch metric history to JSON\n",
            "\n",
            "Network Overrides:\n",
            "  -l, --net-learning-rate <FLOAT>    Override config learning_rate\n",
            "  -e, --net-epochs <INT>             Override config epochs\n",
            "  -b, --net-batch-size <INT>         Override config batch_size\n",
            "\n",
            "Defaults:\n",
            "  --dataset {}\n",
            "  --config  {}"
        ),
        binary_name,
        default_dataset_path(),
        default_config_path()
    )
}

pub fn parse_cli_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut args = std::env::args();
    let binary_name = args.next().unwrap_or_else(|| "mlp".to_string());

    let mut dataset_path = default_dataset_path();
    let mut config_path = default_config_path();
    let mut verbose = false;
    let mut gui = false;
    let mut monitor_options = MonitorOptions::default();
    let mut net_overrides = NetOverrides::default();

    let mut pending_flag: Option<String> = None;
    for arg in args {
        if let Some(flag) = pending_flag.take() {
            match flag.as_str() {
                "--dataset" | "-d" => dataset_path = arg,
                "--config" | "-c" => config_path = arg,
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
        dataset_path,
        config_path,
        verbose,
        gui,
        monitor_options,
        net_overrides,
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
