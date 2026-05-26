use std::error::Error;

use mlp::data::loader::{Dataset, load_dataset};
use mlp::network::callbacks::{Callback, ProgressLogger};
use mlp::network::config::{LayerGroup, NetworkConfig};
use mlp::network::{activation::ActivationFunction, initializer::WeightInitializer};
use mlp::training::{loss::LossFunction, optimizer::OptimizerType};
use mlp::visualization::live_monitor::LiveTrainingMonitorCallback;
use ndarray::{Array2, Axis, s};

struct CliArgs {
    dataset_path: String,
    config_path: String,
    verbose: bool,
    net_overrides: NetOverrides,
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
                "Usage: {binary_name} [--dataset <path>] [--config <path>] [--verbose]\n\
         Defaults:\n\
                     --dataset {}\n\
                     --config  {}\n\
                 Flags:\n\
                     --verbose, -v  Print loaded config in a visual summary\n\
                 Network Overrides:\n\
                     --net-learning-rate <float>  Override config learning_rate\n\
                     --net-epochs <int>           Override config epochs\n\
                     --net-batch-size <int>       Override config batch_size",
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
    let mut net_overrides = NetOverrides::default();

    let mut pending_flag: Option<String> = None;
    for arg in args {
        if let Some(flag) = pending_flag.take() {
            match flag.as_str() {
                "--dataset" | "-d" => dataset_path = arg,
                "--config" | "-c" => config_path = arg,
                "--net-learning-rate" => {
                    net_overrides.learning_rate = Some(
                        arg.parse::<f64>()
                            .map_err(|_| format!("Invalid value for --net-learning-rate: {arg}"))?,
                    );
                }
                "--net-epochs" => {
                    net_overrides.epochs = Some(
                        arg.parse::<usize>()
                            .map_err(|_| format!("Invalid value for --net-epochs: {arg}"))?,
                    );
                }
                "--net-batch-size" => {
                    net_overrides.batch_size = Some(
                        arg.parse::<usize>()
                            .map_err(|_| format!("Invalid value for --net-batch-size: {arg}"))?,
                    );
                }
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
            | "--net-epochs"
            | "--net-batch-size" => pending_flag = Some(arg),
            "--verbose" | "-v" => verbose = true,
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
    println!("+------------------------------------------------------------+");
    println!("|                 MLP LOADED CONFIG SUMMARY                  |");
    println!("+------------------------------------------------------------+");
    println!(" Dataset path : {}", dataset_path);
    println!(" Config path  : {}", config_path);
    println!(" Learning rate: {:.6}", config.learning_rate);
    println!(" Epochs       : {}", config.epochs);
    println!(" Batch size   : {}", config.batch_size);

    let input_sizes: Vec<usize> = config.input_layers.iter().map(|layer| layer.size).collect();
    let hidden_sizes: Vec<usize> = config.hidden_layers.iter().map(|layer| layer.size).collect();
    let output_sizes: Vec<usize> = config.output_layers.iter().map(|layer| layer.size).collect();
    println!(" Input layers : {:?}", input_sizes);
    println!(" Hidden layers: {:?}", hidden_sizes);
    println!(" Output layers: {:?}", output_sizes);

    println!("--------------------------------------------------------------");
    println!(" Resolved transitions (with defaults):");

    for (idx, spec) in config.resolved_layer_specs().iter().enumerate() {
        println!(
            "  {:>2}. {:>3} -> {:>3} | to={} | activation={} | initializer={}",
            idx + 1,
            spec.from_size,
            spec.to_size,
            group_label(spec.to_group),
            activation_label(spec.activation),
            initializer_label(spec.initializer)
        );
    }

    println!("+------------------------------------------------------------+");
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

    let mut monitor = LiveTrainingMonitorCallback::from_env();
    let mut progress_logger = ProgressLogger::new(epochs);
    let mut callbacks: Vec<&mut dyn Callback> = vec![&mut monitor, &mut progress_logger];

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
        "Training summary: train_loss={:.4}, val_loss={:.4}, train_acc={:.4}, val_acc={:.4}",
        metrics.train_loss, metrics.val_loss, metrics.train_accuracy, metrics.val_accuracy
    );
    println!(
        "Monitor collected {} epoch points",
        monitor.history_len()
    );

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
        "Loaded dataset from {}: rows={}, cols={}",
        cli_args.dataset_path,
        dataset.features.nrows(),
        dataset.features.ncols()
    );
    println!(
        "Loaded network config from {}: learning_rate={}, epochs={}, batch_size={}, layers={}",
        cli_args.config_path,
        network.learning_rate,
        network_config.epochs,
        network_config.batch_size,
        network.layers.len()
    );

    train_from_dataset(&dataset, &network_config)?;

    Ok(())
}
