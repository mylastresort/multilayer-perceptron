use mlp::app::cli::{
    apply_net_overrides, default_config_path, default_dataset_path, parse_args, usage,
};
use mlp::app::types::{NetOverrides, Subcommand};
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
    assert!(matches!(cli.subcommand, Subcommand::Train));
}

#[test]
fn parse_args_split_subcommand() {
    let args = ss(&["split"]);
    let cli = parse_args("mlp", &args).unwrap();
    assert!(matches!(cli.subcommand, Subcommand::Split));
}

#[test]
fn parse_args_predict_subcommand() {
    let args = ss(&["predict"]);
    let cli = parse_args("mlp", &args).unwrap();
    assert!(matches!(cli.subcommand, Subcommand::Predict));
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
fn parse_args_early_stopping_defaults_to_false() {
    let args = ss(&[
        "train",
        "--dataset",
        "/tmp/data.csv",
        "--config",
        "/tmp/cfg.yaml",
    ]);
    let cli = parse_args("mlp", &args).unwrap();
    assert!(!cli.monitor_options.early_stopping);
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
