use std::error::Error;

use super::types::{CliArgs, MonitorOptions, NetOverrides, Subcommand};
use crate::network::config::NetworkConfig;
use crate::training::monitor::{MonitorMode, MonitoredMetric, parse_monitored_metrics};
use clap::{CommandFactory, Parser, Subcommand as ClapSubcommand};

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

    #[arg(long = "curves-out", global = true, value_name = "PATH")]
    curves_out: Option<String>,

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
        early_stopping: raw.early_stopping && !raw.no_early_stopping,
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
        curves_out: raw.curves_out.clone(),
        metrics: monitor_metrics_from_raw(raw, &defaults)?,
    })
}

fn monitor_metrics_from_raw(
    raw: &RawCli,
    defaults: &MonitorOptions,
) -> Result<Vec<MonitoredMetric>, Box<dyn Error>> {
    match &raw.monitor_metrics {
        Some(value) => parse_monitored_metrics(value).map_err(Into::into),
        None => Ok(defaults.metrics.clone()),
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
        monitor_options,
        net_overrides,
        model_out: raw.model_out,
        model_in: raw.model_in,
        split_ratio: raw.split_ratio,
        train_out: raw.train_out,
        val_out: raw.val_out,
    })
}

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
