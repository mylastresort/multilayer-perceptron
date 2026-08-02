use std::error::Error;

use clap::{CommandFactory, Parser, Subcommand as ClapSubcommand};
use mlp::network::config::NetworkConfig;
use mlp::training::monitor::{MonitorMode, MonitoredMetric, parse_monitored_metrics};

use super::types::{CliArgs, MonitorOptions, NetOverrides, Subcommand};

pub fn default_dataset_path() -> String {
    format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"))
}

pub fn default_config_path() -> String {
    format!("{}/models/mandatory_sgd.yaml", env!("CARGO_MANIFEST_DIR"))
}

#[derive(Parser)]
#[command(name = "mlp", disable_help_subcommand = true)]
struct RawCli {
    #[arg(short = 'd', long = "dataset", global = true, value_name = "PATH")]
    dataset_path: Option<String>,

    #[arg(short = 'c', long = "config", global = true, value_name = "PATH")]
    config_path: Option<String>,

    #[arg(long = "model-out", global = true, value_name = "PATH")]
    model_out: Option<String>,

    #[arg(long = "model", global = true, value_name = "PATH")]
    model_in: Option<String>,

    #[arg(
        long = "ratio",
        global = true,
        default_value_t = 0.8,
        value_name = "FLOAT"
    )]
    split_ratio: f64,

    #[arg(long = "train-out", global = true, value_name = "PATH")]
    train_out: Option<String>,

    #[arg(long = "val-out", global = true, value_name = "PATH")]
    val_out: Option<String>,

    #[arg(short = 'v', long = "verbose", global = true)]
    verbose: bool,

    #[arg(short = 'g', long = "gui", global = true)]
    gui: bool,

    #[arg(short = 'M', long = "monitor-metrics", global = true)]
    monitor_metrics: Option<String>,

    #[arg(short = 'm', long = "early-stop-metric", global = true, value_parser = parse_metric)]
    early_stop_metric: Option<MonitoredMetric>,

    #[arg(long = "early-stop-mode", global = true, value_parser = parse_monitor_mode)]
    early_stop_mode: Option<MonitorMode>,

    #[arg(short = 'p', long = "early-stop-patience", global = true)]
    early_stop_patience: Option<usize>,

    #[arg(long = "early-stop-min-delta", global = true)]
    early_stop_min_delta: Option<f64>,

    #[arg(short = 's', long = "early-stop-start-epoch", global = true)]
    early_stop_start_epoch: Option<usize>,

    #[arg(long = "early-stopping", global = true)]
    early_stopping: bool,

    #[arg(long = "no-early-stopping", global = true)]
    no_early_stopping: bool,

    #[arg(long = "monitor-history-out", global = true, value_name = "PATH")]
    monitor_history_out: Option<String>,

    #[arg(short = 'l', long = "net-learning-rate", global = true)]
    net_learning_rate: Option<f64>,

    #[arg(short = 'e', long = "net-epochs", global = true)]
    net_epochs: Option<usize>,

    #[arg(short = 'b', long = "net-batch-size", global = true)]
    net_batch_size: Option<usize>,

    #[command(subcommand)]
    command: RawSubcommand,
}

#[derive(ClapSubcommand)]
#[command(rename_all = "kebab-case")]
enum RawSubcommand {
    Split,
    Train,
    Predict,
}

pub fn usage(binary_name: &str) -> String {
    let mut command = RawCli::command().bin_name(binary_name.to_string());
    command.render_long_help().to_string()
}

fn parse_metric(value: &str) -> Result<MonitoredMetric, String> {
    MonitoredMetric::parse(value).ok_or_else(|| format!("Invalid value: {value}"))
}

fn parse_monitor_mode(value: &str) -> Result<MonitorMode, String> {
    MonitorMode::parse(value).ok_or_else(|| format!("Invalid value: {value}"))
}

fn app_subcommand(command: &RawSubcommand) -> Subcommand {
    match command {
        RawSubcommand::Split => Subcommand::Split,
        RawSubcommand::Train => Subcommand::Train,
        RawSubcommand::Predict => Subcommand::Predict,
    }
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

fn resolve_dataset_path(
    subcommand: &Subcommand,
    value: Option<String>,
    binary_name: &str,
) -> Result<String, Box<dyn Error>> {
    match subcommand {
        Subcommand::Train => require_train_path(value, "--dataset", "dataset", binary_name),
        _ => Ok(value.unwrap_or_else(default_dataset_path)),
    }
}

fn resolve_config_path(
    subcommand: &Subcommand,
    value: Option<String>,
    binary_name: &str,
) -> Result<String, Box<dyn Error>> {
    match subcommand {
        Subcommand::Train => require_train_path(value, "--config", "config", binary_name),
        _ => Ok(value.unwrap_or_else(default_config_path)),
    }
}

fn monitor_options_from_raw(raw: &RawCli) -> Result<MonitorOptions, Box<dyn Error>> {
    let defaults = MonitorOptions::default();
    Ok(MonitorOptions {
        early_stopping: raw.early_stopping || !raw.no_early_stopping,
        early_stop_metric: raw.early_stop_metric.unwrap_or(defaults.early_stop_metric),
        early_stop_mode: raw.early_stop_mode.unwrap_or(defaults.early_stop_mode),
        early_stop_patience: raw
            .early_stop_patience
            .unwrap_or(defaults.early_stop_patience),
        early_stop_min_delta: raw
            .early_stop_min_delta
            .unwrap_or(defaults.early_stop_min_delta),
        early_stop_start_epoch: raw
            .early_stop_start_epoch
            .unwrap_or(defaults.early_stop_start_epoch),
        history_out: raw.monitor_history_out.clone(),
        metrics: monitor_metrics_from_raw(raw)?,
    })
}

fn monitor_metrics_from_raw(raw: &RawCli) -> Result<Vec<MonitoredMetric>, Box<dyn Error>> {
    match &raw.monitor_metrics {
        Some(value) => parse_monitored_metrics(value).map_err(Into::into),
        None => Ok(Vec::new()),
    }
}

fn net_overrides_from_raw(raw: &RawCli) -> NetOverrides {
    NetOverrides {
        learning_rate: raw.net_learning_rate,
        epochs: raw.net_epochs,
        batch_size: raw.net_batch_size,
    }
}

fn resolve_args(binary_name: &str, raw: RawCli) -> Result<CliArgs, Box<dyn Error>> {
    let subcommand = app_subcommand(&raw.command);
    let monitor_options = monitor_options_from_raw(&raw)?;
    let net_overrides = net_overrides_from_raw(&raw);
    let dataset_path = resolve_dataset_path(&subcommand, raw.dataset_path, binary_name)?;
    let config_path = resolve_config_path(&subcommand, raw.config_path, binary_name)?;

    Ok(CliArgs {
        subcommand,
        dataset_path,
        config_path,
        verbose: raw.verbose,
        gui: raw.gui,
        monitor_options,
        net_overrides,
        model_out: raw.model_out,
        model_in: raw.model_in,
        split_ratio: raw.split_ratio,
        train_out: raw.train_out,
        val_out: raw.val_out,
    })
}

#[cfg(test)]
pub fn parse_args(binary_name: &str, rest: &[String]) -> Result<CliArgs, Box<dyn Error>> {
    if rest.is_empty() {
        return Err(usage(binary_name).into());
    }

    let args = std::iter::once(binary_name.to_string())
        .chain(rest.iter().cloned())
        .collect::<Vec<_>>();
    let raw = RawCli::try_parse_from(args).map_err(|err| err.to_string())?;
    resolve_args(binary_name, raw)
}

pub fn parse_env_args() -> Result<CliArgs, Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let binary_name = args.first().cloned().unwrap_or_else(|| "mlp".to_string());
    if args.len() == 1 {
        println!("{}", usage(&binary_name));
        std::process::exit(0);
    }
    let raw = RawCli::parse_from(args);
    resolve_args(&binary_name, raw)
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
