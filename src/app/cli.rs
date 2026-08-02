use std::error::Error;

use mlp::network::config::NetworkConfig;
use mlp::training::monitor::{MonitorMode, MonitoredMetric, parse_monitored_metrics};

use super::types::{CliArgs, MonitorOptions, NetOverrides, Subcommand};

pub fn default_dataset_path() -> String {
    format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"))
}

pub fn default_config_path() -> String {
    format!("{}/models/mandatory_sgd.yaml", env!("CARGO_MANIFEST_DIR"))
}

const USAGE_TEMPLATE: &str = concat!(
    "Usage: {bin} <SUBCOMMAND> [OPTIONS]\n",
    "\n",
    "Subcommands:\n",
    "  split    Separate the dataset into train and validation CSV files\n",
    "  train    Train the network and save the model\n",
    "  predict  Load a saved model and evaluate on a dataset (binary cross-entropy)\n",
    "\n",
    "Common options:\n",
    "  -d, --dataset <PATH>               Dataset CSV path (required for train)\n",
    "  -h, --help                         Print help\n",
    "\n",
    "split options:\n",
    "      --train-out <PATH>             Output CSV for training split\n",
    "      --val-out   <PATH>             Output CSV for validation split\n",
    "      --ratio     <FLOAT>            Training fraction (default 0.8)\n",
    "\n",
    "train options:\n",
    "  -d, --dataset <PATH>               Dataset CSV path (required)\n",
    "  -c, --config <PATH>                Network/training YAML config path (required)\n",
    "      --model-out <PATH>             Where to save the trained model (JSON)\n",
    "  -v, --verbose                      Print loaded config summary\n",
    "  -g, --gui                          Open live learning-curve GUI\n",
    "  -M, --monitor-metrics <CSV>        Metrics to monitor/plot\n",
    "  -m, --early-stop-metric <METRIC>    Metric that drives early stopping\n",
    "      --early-stop-mode <MODE>        Early-stopping direction [min|max]\n",
    "  -p, --early-stop-patience <INT>     Early-stopping patience in epochs\n",
    "      --early-stop-min-delta <FLOAT>  Minimum improvement threshold\n",
    "  -s, --early-stop-start-epoch <INT>  Epoch to start early-stopping checks\n",
    "      --early-stopping                Enable early stopping (on by default)\n",
    "      --no-early-stopping            Disable early stopping\n",
    "      --monitor-history-out <PATH>   Save per-epoch metric history to JSON\n",
    "  -l, --net-learning-rate <FLOAT>    Override config learning_rate\n",
    "  -e, --net-epochs <INT>             Override config epochs\n",
    "  -b, --net-batch-size <INT>         Override config batch_size\n",
    "\n",
    "predict options:\n",
    "      --model <PATH>                 Path to a saved model JSON\n",
    "\n",
    "Defaults (split/predict):\n",
    "  --dataset {dataset}\n",
    "  --config  {config}",
);

pub fn usage(binary_name: &str) -> String {
    USAGE_TEMPLATE
        .replace("{bin}", binary_name)
        .replace("{dataset}", &default_dataset_path())
        .replace("{config}", &default_config_path())
}

const VALUE_FLAGS: &[&str] = &[
    "--dataset",
    "-d",
    "--config",
    "-c",
    "--model-out",
    "--model",
    "--train-out",
    "--val-out",
    "--ratio",
    "--net-learning-rate",
    "-l",
    "--net-epochs",
    "-e",
    "--net-batch-size",
    "-b",
    "--monitor-metrics",
    "-M",
    "--early-stop-metric",
    "-m",
    "--early-stop-mode",
    "--early-stop-patience",
    "-p",
    "--early-stop-min-delta",
    "--early-stop-start-epoch",
    "-s",
    "--monitor-history-out",
];

#[derive(Default)]
struct ParsedFlags {
    dataset_path: Option<String>,
    config_path: Option<String>,
    model_out: Option<String>,
    model_in: Option<String>,
    split_ratio: f64,
    train_out: Option<String>,
    val_out: Option<String>,
    verbose: bool,
    gui: bool,
    monitor_options: MonitorOptions,
    net_overrides: NetOverrides,
}

enum FlagAction {
    ExpectValue(String),
    Handled,
}

fn parse_value<T>(arg: &str, flag: &str) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
{
    arg.parse::<T>()
        .map_err(|_| format!("Invalid value for {flag}: {arg}").into())
}

fn parse_subcommand(binary_name: &str, first: &str) -> Result<Subcommand, Box<dyn Error>> {
    match first {
        "split" => Ok(Subcommand::Split),
        "train" => Ok(Subcommand::Train),
        "predict" => Ok(Subcommand::Predict),
        other => Err(format!("Unknown subcommand: '{other}'\n{}", usage(binary_name)).into()),
    }
}

fn apply_flag(
    flags: &mut ParsedFlags,
    arg: &str,
    binary_name: &str,
) -> Result<FlagAction, Box<dyn Error>> {
    if VALUE_FLAGS.contains(&arg) {
        return Ok(FlagAction::ExpectValue(arg.to_string()));
    }
    match arg {
        "--verbose" | "-v" => flags.verbose = true,
        "--gui" | "-g" => flags.gui = true,
        "--early-stopping" => flags.monitor_options.early_stopping = true,
        "--no-early-stopping" => flags.monitor_options.early_stopping = false,
        "--help" | "-h" => return Err(usage(binary_name).into()),
        _ => return Err(format!("Unknown argument: {arg}\n{}", usage(binary_name)).into()),
    }
    Ok(FlagAction::Handled)
}

fn parse_net_numeric_flags(
    flags: &mut ParsedFlags,
    flag: &str,
    arg: &str,
) -> Result<bool, Box<dyn Error>> {
    Ok(match flag {
        "--net-learning-rate" | "-l" => {
            flags.net_overrides.learning_rate = Some(parse_value(arg, flag)?);
            true
        }
        "--net-epochs" | "-e" => {
            flags.net_overrides.epochs = Some(parse_value(arg, flag)?);
            true
        }
        "--net-batch-size" | "-b" => {
            flags.net_overrides.batch_size = Some(parse_value(arg, flag)?);
            true
        }
        _ => false,
    })
}

fn parse_monitor_numeric_flags(
    flags: &mut ParsedFlags,
    flag: &str,
    arg: &str,
) -> Result<bool, Box<dyn Error>> {
    Ok(match flag {
        "--early-stop-patience" | "-p" => {
            flags.monitor_options.early_stop_patience = parse_value(arg, flag)?;
            true
        }
        "--early-stop-min-delta" => {
            flags.monitor_options.early_stop_min_delta = parse_value(arg, flag)?;
            true
        }
        "--early-stop-start-epoch" | "-s" => {
            flags.monitor_options.early_stop_start_epoch = parse_value(arg, flag)?;
            true
        }
        _ => false,
    })
}

fn parse_numeric_flags(
    flags: &mut ParsedFlags,
    flag: &str,
    arg: &str,
) -> Result<bool, Box<dyn Error>> {
    if flag == "--ratio" {
        flags.split_ratio = parse_value(arg, flag)?;
        return Ok(true);
    }
    if parse_net_numeric_flags(flags, flag, arg)? {
        return Ok(true);
    }
    parse_monitor_numeric_flags(flags, flag, arg)
}

fn apply_value_flag(
    flags: &mut ParsedFlags,
    flag: &str,
    arg: String,
) -> Result<(), Box<dyn Error>> {
    if parse_numeric_flags(flags, flag, &arg)? {
        return Ok(());
    }
    match flag {
        "--dataset" | "-d" => flags.dataset_path = Some(arg),
        "--config" | "-c" => flags.config_path = Some(arg),
        "--model-out" => flags.model_out = Some(arg),
        "--model" => flags.model_in = Some(arg),
        "--train-out" => flags.train_out = Some(arg),
        "--val-out" => flags.val_out = Some(arg),
        "--monitor-metrics" | "-M" => {
            flags.monitor_options.metrics = parse_monitored_metrics(&arg)?;
        }
        "--early-stop-metric" | "-m" => {
            flags.monitor_options.early_stop_metric = MonitoredMetric::parse(&arg)
                .ok_or_else(|| format!("Invalid value for {flag}: {arg}"))?;
        }
        "--early-stop-mode" => {
            flags.monitor_options.early_stop_mode = MonitorMode::parse(&arg)
                .ok_or_else(|| format!("Invalid value for {flag}: {arg}"))?;
        }
        "--monitor-history-out" => flags.monitor_options.history_out = Some(arg),
        _ => unreachable!("unsupported flag in parser state"),
    }
    Ok(())
}

fn parse_flags(
    binary_name: &str,
    rest: &[String],
    flags: &mut ParsedFlags,
) -> Result<(), Box<dyn Error>> {
    let mut pending_flag: Option<String> = None;
    for arg in rest.iter().skip(1).cloned() {
        if let Some(flag) = pending_flag.take() {
            apply_value_flag(flags, &flag, arg)?;
            continue;
        }
        match apply_flag(flags, &arg, binary_name)? {
            FlagAction::ExpectValue(flag) => pending_flag = Some(flag),
            FlagAction::Handled => {}
        }
    }

    if let Some(flag) = pending_flag {
        return Err(format!("Missing value for {flag}\n{}", usage(binary_name)).into());
    }
    Ok(())
}

fn require_train_path(
    value: Option<String>,
    flag: &str,
    label: &str,
    binary_name: &str,
) -> Result<String, Box<dyn Error>> {
    value.ok_or_else(|| {
        format!(
            "train requires {flag} <PATH>: no default {label} is used for training\n{}",
            usage(binary_name)
        )
        .into()
    })
}

fn resolve_args(
    binary_name: &str,
    subcommand: Subcommand,
    flags: ParsedFlags,
) -> Result<CliArgs, Box<dyn Error>> {
    let dataset_path = match subcommand {
        Subcommand::Train => {
            require_train_path(flags.dataset_path, "--dataset", "dataset", binary_name)?
        }
        _ => flags.dataset_path.unwrap_or_else(default_dataset_path),
    };
    let config_path = match subcommand {
        Subcommand::Train => {
            require_train_path(flags.config_path, "--config", "config", binary_name)?
        }
        _ => flags.config_path.unwrap_or_else(default_config_path),
    };

    Ok(CliArgs {
        subcommand,
        dataset_path,
        config_path,
        verbose: flags.verbose,
        gui: flags.gui,
        monitor_options: flags.monitor_options,
        net_overrides: flags.net_overrides,
        model_out: flags.model_out,
        model_in: flags.model_in,
        split_ratio: flags.split_ratio,
        train_out: flags.train_out,
        val_out: flags.val_out,
    })
}

pub fn parse_args(binary_name: &str, rest: &[String]) -> Result<CliArgs, Box<dyn Error>> {
    if rest.is_empty() || rest[0] == "--help" || rest[0] == "-h" {
        return Err(usage(binary_name).into());
    }

    let subcommand = parse_subcommand(binary_name, &rest[0])?;
    let mut flags = ParsedFlags {
        split_ratio: 0.8,
        ..Default::default()
    };
    parse_flags(binary_name, rest, &mut flags)?;
    resolve_args(binary_name, subcommand, flags)
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
    use super::{
        NetOverrides, apply_net_overrides, default_config_path, default_dataset_path, parse_args,
        usage,
    };
    use mlp::network::config::NetworkConfig;

    fn ss(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

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
    fn parse_args_train_subcommand() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(matches!(cli.subcommand, super::Subcommand::Train));
    }

    #[test]
    fn parse_args_split_subcommand() {
        let args = ss(&["split"]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(matches!(cli.subcommand, super::Subcommand::Split));
    }

    #[test]
    fn parse_args_predict_subcommand() {
        let args = ss(&["predict"]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(matches!(cli.subcommand, super::Subcommand::Predict));
    }

    #[test]
    fn parse_args_empty_returns_err() {
        assert!(parse_args("mlp", &[]).is_err());
    }

    #[test]
    fn parse_args_help_long_flag_returns_err() {
        assert!(parse_args("mlp", &ss(&["--help"])).is_err());
    }

    #[test]
    fn parse_args_help_short_flag_returns_err() {
        assert!(parse_args("mlp", &ss(&["-h"])).is_err());
    }

    #[test]
    fn parse_args_unknown_subcommand_returns_err() {
        assert!(parse_args("mlp", &ss(&["unknown"])).is_err());
    }

    #[test]
    fn parse_args_dataset_long_flag() {
        let args = ss(&[
            "train",
            "--config",
            "/tmp/cfg.yaml",
            "--dataset",
            "/tmp/data.csv",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.dataset_path, "/tmp/data.csv");
    }

    #[test]
    fn parse_args_dataset_short_flag() {
        let args = ss(&["train", "--config", "/tmp/cfg.yaml", "-d", "/tmp/data.csv"]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.dataset_path, "/tmp/data.csv");
    }

    #[test]
    fn parse_args_config_long_flag() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.config_path, "/tmp/cfg.yaml");
    }

    #[test]
    fn parse_args_config_short_flag() {
        let args = ss(&["train", "--dataset", "/tmp/data.csv", "-c", "/tmp/cfg.yaml"]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.config_path, "/tmp/cfg.yaml");
    }

    #[test]
    fn parse_args_verbose_long_flag() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--verbose",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn parse_args_verbose_short_flag() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-v",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(cli.verbose);
    }

    #[test]
    fn parse_args_gui_long_flag() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--gui",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(cli.gui);
    }

    #[test]
    fn parse_args_gui_short_flag() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-g",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(cli.gui);
    }

    #[test]
    fn parse_args_split_train_out_and_val_out() {
        let args = ss(&[
            "split",
            "--train-out",
            "/tmp/train.csv",
            "--val-out",
            "/tmp/val.csv",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.train_out.as_deref(), Some("/tmp/train.csv"));
        assert_eq!(cli.val_out.as_deref(), Some("/tmp/val.csv"));
    }

    #[test]
    fn parse_args_split_ratio() {
        let args = ss(&["split", "--ratio", "0.75"]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!((cli.split_ratio - 0.75).abs() < 1e-12);
    }

    #[test]
    fn parse_args_split_ratio_invalid_returns_err() {
        let args = ss(&["split", "--ratio", "not_a_float"]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_model_out() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--model-out",
            "/tmp/model.json",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.model_out.as_deref(), Some("/tmp/model.json"));
    }

    #[test]
    fn parse_args_net_learning_rate_long() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--net-learning-rate",
            "0.001",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!((cli.net_overrides.learning_rate.unwrap() - 0.001).abs() < 1e-12);
    }

    #[test]
    fn parse_args_net_learning_rate_short() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-l",
            "0.002",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!((cli.net_overrides.learning_rate.unwrap() - 0.002).abs() < 1e-12);
    }

    #[test]
    fn parse_args_net_learning_rate_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--net-learning-rate",
            "bad",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_net_epochs_long() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--net-epochs",
            "50",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.net_overrides.epochs, Some(50));
    }

    #[test]
    fn parse_args_net_epochs_short() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-e",
            "20",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.net_overrides.epochs, Some(20));
    }

    #[test]
    fn parse_args_net_epochs_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--net-epochs",
            "bad",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_net_batch_size_long() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--net-batch-size",
            "32",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.net_overrides.batch_size, Some(32));
    }

    #[test]
    fn parse_args_net_batch_size_short() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-b",
            "16",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.net_overrides.batch_size, Some(16));
    }

    #[test]
    fn parse_args_net_batch_size_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--net-batch-size",
            "bad",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_early_stopping_flag() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stopping",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(cli.monitor_options.early_stopping);
    }

    #[test]
    fn parse_args_no_early_stopping_flag() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--no-early-stopping",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(!cli.monitor_options.early_stopping);
    }

    #[test]
    fn parse_args_early_stopping_defaults_to_true() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(cli.monitor_options.early_stopping);
    }

    #[test]
    fn parse_args_early_stop_patience_long() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-patience",
            "5",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.monitor_options.early_stop_patience, 5);
    }

    #[test]
    fn parse_args_early_stop_patience_short() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-p",
            "3",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.monitor_options.early_stop_patience, 3);
    }

    #[test]
    fn parse_args_early_stop_patience_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-patience",
            "bad",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_early_stop_min_delta() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-min-delta",
            "0.001",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!((cli.monitor_options.early_stop_min_delta - 0.001).abs() < 1e-12);
    }

    #[test]
    fn parse_args_early_stop_min_delta_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-min-delta",
            "bad",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_early_stop_start_epoch_long() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-start-epoch",
            "10",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.monitor_options.early_stop_start_epoch, 10);
    }

    #[test]
    fn parse_args_early_stop_start_epoch_short() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-s",
            "5",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.monitor_options.early_stop_start_epoch, 5);
    }

    #[test]
    fn parse_args_early_stop_start_epoch_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-start-epoch",
            "bad",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_monitor_history_out() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--monitor-history-out",
            "/tmp/history.json",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(
            cli.monitor_options.history_out.as_deref(),
            Some("/tmp/history.json")
        );
    }

    #[test]
    fn parse_args_early_stop_mode_valid() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-mode",
            "min",
        ]);
        let cli = parse_args("mlp", &args).unwrap();
        assert!(matches!(
            cli.monitor_options.early_stop_mode,
            mlp::training::monitor::MonitorMode::Min
        ));
    }

    #[test]
    fn parse_args_early_stop_mode_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-mode",
            "bad",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_early_stop_metric_valid() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-metric",
            "loss",
        ]);
        assert!(parse_args("mlp", &args).is_ok());
    }

    #[test]
    fn parse_args_early_stop_metric_short() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-m",
            "loss",
        ]);
        assert!(parse_args("mlp", &args).is_ok());
    }

    #[test]
    fn parse_args_early_stop_metric_invalid_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--early-stop-metric",
            "unknown_metric_xyz",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_monitor_metrics_long() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--monitor-metrics",
            "loss,accuracy",
        ]);
        assert!(parse_args("mlp", &args).is_ok());
    }

    #[test]
    fn parse_args_monitor_metrics_short() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-M",
            "loss",
        ]);
        assert!(parse_args("mlp", &args).is_ok());
    }

    #[test]
    fn parse_args_predict_model_flag() {
        let args = ss(&["predict", "--model", "/tmp/model.json"]);
        let cli = parse_args("mlp", &args).unwrap();
        assert_eq!(cli.model_in.as_deref(), Some("/tmp/model.json"));
    }

    #[test]
    fn parse_args_unknown_argument_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--unknown-flag",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_help_inside_subcommand_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "--help",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_help_short_inside_subcommand_returns_err() {
        let args = ss(&[
            "train",
            "--dataset",
            "/tmp/data.csv",
            "--config",
            "/tmp/cfg.yaml",
            "-h",
        ]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_missing_value_for_flag_returns_err() {
        let args = ss(&["train", "--config", "/tmp/cfg.yaml", "--dataset"]);
        assert!(parse_args("mlp", &args).is_err());
    }

    #[test]
    fn parse_args_defaults_are_set() {
        let cli = parse_args("mlp", &ss(&["split"])).unwrap();
        assert!(cli.dataset_path.ends_with("data.csv"));
        assert!(cli.config_path.ends_with(".yaml"));
        assert!(!cli.verbose);
        assert!((cli.split_ratio - 0.8).abs() < 1e-12);
        assert!(cli.model_out.is_none());
        assert!(cli.model_in.is_none());
    }

    #[test]
    fn apply_net_overrides_sets_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides {
            learning_rate: Some(0.001),
            ..NetOverrides::default()
        };
        apply_net_overrides(&mut config, &overrides).unwrap();
        assert!((config.learning_rate - 0.001).abs() < 1e-12);
    }

    #[test]
    fn apply_net_overrides_sets_epochs() {
        let mut config = make_config();
        let overrides = NetOverrides {
            epochs: Some(5),
            ..NetOverrides::default()
        };
        apply_net_overrides(&mut config, &overrides).unwrap();
        assert_eq!(config.epochs, 5);
    }

    #[test]
    fn apply_net_overrides_sets_batch_size() {
        let mut config = make_config();
        let overrides = NetOverrides {
            batch_size: Some(16),
            ..NetOverrides::default()
        };
        apply_net_overrides(&mut config, &overrides).unwrap();
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn apply_net_overrides_rejects_zero_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides {
            learning_rate: Some(0.0),
            ..NetOverrides::default()
        };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_negative_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides {
            learning_rate: Some(-0.01),
            ..NetOverrides::default()
        };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_infinite_learning_rate() {
        let mut config = make_config();
        let overrides = NetOverrides {
            learning_rate: Some(f64::INFINITY),
            ..NetOverrides::default()
        };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_zero_epochs() {
        let mut config = make_config();
        let overrides = NetOverrides {
            epochs: Some(0),
            ..NetOverrides::default()
        };
        assert!(apply_net_overrides(&mut config, &overrides).is_err());
    }

    #[test]
    fn apply_net_overrides_rejects_zero_batch_size() {
        let mut config = make_config();
        let overrides = NetOverrides {
            batch_size: Some(0),
            ..NetOverrides::default()
        };
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
