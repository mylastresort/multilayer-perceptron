use std::error::Error;

use mlp::console::{Tone, bold, paint};
use mlp::data::loader::{Dataset, load_dataset};
use mlp::network::callbacks::{Callback, ProgressLogger};
use mlp::network::config::{LayerGroup, NetworkConfig};
use mlp::network::{activation::ActivationFunction, initializer::WeightInitializer};
use mlp::training::monitor::{
    EarlyStoppingCallback, EarlyStoppingConfig, HistoryCallback, MonitorMode, MonitoredMetric,
    parse_monitored_metrics,
};
use mlp::training::{loss::LossFunction, optimizer::OptimizerType};
use mlp::visualization::live_monitor::{GuiMonitorConfig, LiveTrainingMonitorCallback};
use ndarray::{Array2, Axis, s};

struct CliArgs {
    dataset_path: String,
    config_path: String,
    verbose: bool,
    gui: bool,
    monitor_options: MonitorOptions,
    net_overrides: NetOverrides,
}

struct MonitorOptions {
    metrics: Vec<MonitoredMetric>,
    early_stopping: bool,
    monitor_metric: MonitoredMetric,
    monitor_mode: MonitorMode,
    monitor_patience: usize,
    monitor_min_delta: f64,
    monitor_start_epoch: usize,
    history_out: Option<String>,
}

impl Default for MonitorOptions {
    fn default() -> Self {
        Self {
            metrics: vec![MonitoredMetric::Loss, MonitoredMetric::Accuracy],
            early_stopping: false,
            monitor_metric: MonitoredMetric::Loss,
            monitor_mode: MonitorMode::Min,
            monitor_patience: 10,
            monitor_min_delta: 0.0,
            monitor_start_epoch: 0,
            history_out: None,
        }
    }
}

#[derive(Default)]
struct NetOverrides {
    learning_rate: Option<f64>,
    epochs: Option<usize>,
    batch_size: Option<usize>,
}

fn default_dataset_path() -> String {
    format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"))
}

fn default_config_path() -> String {
    format!(
        "{}/models/training_learning_curve.yaml",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn usage(binary_name: &str) -> String {
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

fn parse_cli_args() -> Result<CliArgs, Box<dyn Error>> {
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
                    net_overrides.learning_rate = Some(
                        arg.parse::<f64>()
                            .map_err(|_| format!("Invalid value for --net-learning-rate: {arg}"))?,
                    );
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

fn apply_net_overrides(config: &mut NetworkConfig, overrides: &NetOverrides) -> Result<(), Box<dyn Error>> {
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

fn activation_label(activation: ActivationFunction) -> &'static str {
    match activation {
        ActivationFunction::Sigmoid => "sigmoid",
        ActivationFunction::Tanh => "tanh",
        ActivationFunction::ReLU => "relu",
        ActivationFunction::Softmax => "softmax",
    }
}

fn initializer_label(initializer: WeightInitializer) -> &'static str {
    match initializer {
        WeightInitializer::Random => "random",
        WeightInitializer::Xavier => "xavier",
        WeightInitializer::He => "he",
    }
}

fn group_label(group: LayerGroup) -> &'static str {
    match group {
        LayerGroup::Input => "input",
        LayerGroup::Hidden => "hidden",
        LayerGroup::Output => "output",
    }
}

fn print_verbose_config(config: &NetworkConfig, dataset_path: &str, config_path: &str) {
    println!("{}", paint("+------------------------------------------------------------+", Tone::Accent));
    println!("{}", bold(&paint("|                 MLP LOADED CONFIG SUMMARY                  |", Tone::Accent)));
    println!("{}", paint("+------------------------------------------------------------+", Tone::Accent));
    println!(" {} {}", paint("Dataset path :", Tone::Info), dataset_path);
    println!(" {} {}", paint("Config path  :", Tone::Info), config_path);
    println!(" {} {:.6}", paint("Learning rate:", Tone::Info), config.learning_rate);
    println!(" {} {}", paint("Epochs       :", Tone::Info), config.epochs);
    println!(" {} {}", paint("Batch size   :", Tone::Info), config.batch_size);

    let input_sizes: Vec<usize> = config.input_layers.iter().map(|layer| layer.size).collect();
    let hidden_sizes: Vec<usize> = config.hidden_layers.iter().map(|layer| layer.size).collect();
    let output_sizes: Vec<usize> = config.output_layers.iter().map(|layer| layer.size).collect();
    println!(" {} {:?}", paint("Input layers :", Tone::TrainMetric), input_sizes);
    println!(" {} {:?}", paint("Hidden layers:", Tone::TrainMetric), hidden_sizes);
    println!(" {} {:?}", paint("Output layers:", Tone::ValMetric), output_sizes);

    println!("{}", paint("--------------------------------------------------------------", Tone::Muted));
    println!(" {}", bold(&paint("Resolved transitions (with defaults):", Tone::Info)));

    for (idx, spec) in config.resolved_layer_specs().iter().enumerate() {
        println!(
            "  {:>2}. {:>3} -> {:>3} | {}={} | {}={} | {}={}",
            idx + 1,
            spec.from_size,
            spec.to_size,
            paint("to", Tone::Muted),
            group_label(spec.to_group),
            paint("activation", Tone::Muted),
            activation_label(spec.activation),
            paint("initializer", Tone::Muted),
            initializer_label(spec.initializer)
        );
    }

    println!("{}", paint("+------------------------------------------------------------+", Tone::Accent));
}

fn build_dataset(dataset_path: &str) -> Result<Dataset, Box<dyn Error>> {
    let base_features = vec![
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ];
    let stats = vec!["mean", "se", "extreme"];

    let mut names: Vec<String> = vec!["ID".to_string(), "Diagnosis".to_string()];
    for feature in &base_features {
        for stat in &stats {
            names.push(format!("{}_{}", feature, stat));
        }
    }

    // Keep loader defaults aligned with training_learning_curve_test.
    load_dataset(dataset_path, 1, names, 0)
}

fn standardize_from_train(x_train: &Array2<f64>, x_other: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
    let means = x_train
        .mean_axis(Axis(0))
        .expect("training features should not be empty");
    let stds = x_train.std_axis(Axis(0), 0.0).mapv(|v| v.max(1e-12));

    let x_train_scaled = (x_train - &means) / &stds;
    let x_other_scaled = (x_other - &means) / &stds;

    (x_train_scaled, x_other_scaled)
}

fn train_from_dataset(
    dataset: &Dataset,
    network_config: &NetworkConfig,
    gui: bool,
    monitor_options: &MonitorOptions,
) -> Result<(), Box<dyn Error>> {
    // Baseline feature prep mirrors training_learning_curve_test.
    let x_raw = dataset.features.slice(s![.., 1..]).to_owned();
    let y = dataset
        .features
        .column(0)
        .mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });

    let n = x_raw.nrows();
    if n < 3 {
        return Err("dataset must contain at least 3 rows to split train/val/test".into());
    }

    let train_end = (0.70 * n as f64).round() as usize;
    let val_end = (0.85 * n as f64).round() as usize;
    if train_end == 0 || val_end <= train_end {
        return Err("dataset split produced empty training or validation set".into());
    }

    let x_train_raw = x_raw.slice(s![0..train_end, ..]).to_owned();
    let x_val_raw = x_raw.slice(s![train_end..val_end, ..]).to_owned();
    let y_train = y.slice(s![0..train_end]).to_owned();
    let y_val = y.slice(s![train_end..val_end]).to_owned();
    let (x_train, x_val) = standardize_from_train(&x_train_raw, &x_val_raw);

    let mut network = network_config.build_network();
    let epochs = network_config.epochs;
    let batch_size = network_config.batch_size;

    let mut monitor = LiveTrainingMonitorCallback::new(GuiMonitorConfig::from_env(
        gui,
        monitor_options.metrics.clone(),
    ));
    let mut history = HistoryCallback::new();
    let mut early_stopping = EarlyStoppingCallback::new(EarlyStoppingConfig {
        enabled: monitor_options.early_stopping,
        metric: monitor_options.monitor_metric,
        mode: monitor_options.monitor_mode,
        patience: monitor_options.monitor_patience,
        min_delta: monitor_options.monitor_min_delta,
        start_epoch: monitor_options.monitor_start_epoch,
    });
    let mut progress_logger = ProgressLogger::new(epochs);
    let mut callbacks: Vec<&mut dyn Callback> = vec![
        &mut monitor,
        &mut history,
        &mut early_stopping,
        &mut progress_logger,
    ];

    let metrics = network.fit_with_callbacks(
        x_train.view(),
        y_train.view(),
        Some((x_val.view(), y_val.view())),
        batch_size,
        epochs,
        OptimizerType::SGD,
        LossFunction::CategoricalCrossEntropy,
        &mut callbacks,
    );

    drop(callbacks);

    println!(
        "{} {} - {} - {} - {}",
        bold(&paint("Training summary:", Tone::Success)),
        paint(&format!("train_loss={:.4}", metrics.train_loss), Tone::TrainMetric),
        paint(&format!("val_loss={:.4}", metrics.val_loss), Tone::ValMetric),
        paint(&format!("train_acc={:.4}", metrics.train_accuracy), Tone::TrainMetric),
        paint(&format!("val_acc={:.4}", metrics.val_accuracy), Tone::ValMetric)
    );
    println!(
        "{} {}",
        paint("Monitor points:", Tone::Info),
        paint(&format!("{} epochs", monitor.history_len()), Tone::Accent)
    );
    println!(
        "{} {}",
        paint("History records:", Tone::Info),
        paint(&format!("{} epochs", history.history().epochs.len()), Tone::Accent)
    );
    if let Some(best_epoch) = early_stopping.best_epoch() {
        println!(
            "{} {}",
            paint("Early stopping best epoch:", Tone::Warn),
            paint(&best_epoch.to_string(), Tone::Warn)
        );
    }
    if early_stopping.stopped() {
        println!(
            "{} {}",
            paint("Early stopping triggered on metric", Tone::Warn),
            paint(monitor_options.monitor_metric.as_str(), Tone::Warn)
        );
    }

    if let Some(path) = &monitor_options.history_out {
        history.save_json(path)?;
        println!(
            "{} {}",
            paint("Saved metric history:", Tone::Success),
            path
        );
    }

    monitor.keep_open_until_closed();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli_args = parse_cli_args()?;
    let dataset = build_dataset(&cli_args.dataset_path)?;
    let mut network_config = NetworkConfig::from_yaml_file(&cli_args.config_path)?;
    apply_net_overrides(&mut network_config, &cli_args.net_overrides)?;
    let network = network_config.build_network();

    if cli_args.verbose {
        print_verbose_config(&network_config, &cli_args.dataset_path, &cli_args.config_path);
    }

    println!(
        "{} {} {}",
        bold(&paint("Loaded dataset:", Tone::Info)),
        cli_args.dataset_path,
        paint(
            &format!("(rows={}, cols={})", dataset.features.nrows(), dataset.features.ncols()),
            Tone::Muted
        )
    );
    println!(
        "{} {} {}",
        bold(&paint("Loaded config:", Tone::Info)),
        cli_args.config_path,
        paint(
            &format!(
                "(learning_rate={}, epochs={}, batch_size={}, layers={})",
                network.learning_rate,
                network_config.epochs,
                network_config.batch_size,
                network.layers.len()
            ),
            Tone::Muted
        )
    );

    train_from_dataset(&dataset, &network_config, cli_args.gui, &cli_args.monitor_options)?;

    Ok(())
}
