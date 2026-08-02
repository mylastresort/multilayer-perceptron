use std::fs;
use std::process::Command;

use mlp::data::loader::load_dataset;
use mlp::network::activation::ActivationFunction;
use mlp::network::callbacks::{Callback, CallbackLogs, ProgressLogger};
use mlp::network::initializer::WeightInitializer;
use mlp::network::layer::Layer;
use mlp::network::model::Network;
use mlp::training::loss::LossFunction;
use mlp::training::optimizer::OptimizerType;
use mlp::visualization::plotter::{
    TrainingHistory, plot_accuracy_curve, plot_loss_curve, plot_training_curves,
};
use ndarray::{Array2, Axis, s};

fn build_dataset_column_names() -> Vec<String> {
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
    names
}

fn maybe_open_plot(output_path: &str) {
    if std::env::var("MLP_OPEN_PLOTS").as_deref() != Ok("1") {
        return;
    }

    if std::env::var("CI").is_ok() {
        return;
    }

    let _ = Command::new("xdg-open").arg(output_path).spawn();
}

fn should_write_png_artifact() -> bool {
    std::env::var("MLP_WRITE_PNG").as_deref() != Ok("0")
}

fn standardize_from_train(
    x_train: &Array2<f64>,
    x_other: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let means = x_train
        .mean_axis(Axis(0))
        .expect("training features should not be empty");
    let stds = x_train.std_axis(Axis(0), 0.0).mapv(|v| v.max(1e-12));

    let x_train_scaled = (x_train - &means) / &stds;
    let x_other_scaled = (x_other - &means) / &stds;

    (x_train_scaled, x_other_scaled)
}

struct MetricsCallback {
    train_losses: Vec<f64>,
    val_losses: Vec<f64>,
    train_accuracies: Vec<f64>,
    val_accuracies: Vec<f64>,
    train_precisions: Vec<f64>,
    val_precisions: Vec<f64>,
}

impl MetricsCallback {
    fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            train_precisions: Vec::new(),
            val_precisions: Vec::new(),
        }
    }
}

impl Callback for MetricsCallback {
    fn on_epoch_end(&mut self, _epoch: usize, logs: Option<&CallbackLogs>) {
        if let Some(logs) = logs {
            if let Some(loss) = logs.loss {
                self.train_losses.push(loss);
            }
            if let Some(val_loss) = logs.val_loss {
                self.val_losses.push(val_loss);
            }
            if let Some(acc) = logs.accuracy {
                self.train_accuracies.push(acc);
            }
            if let Some(val_acc) = logs.val_accuracy {
                self.val_accuracies.push(val_acc);
            }
            if let Some(prec) = logs.precision {
                self.train_precisions.push(prec);
            }
            if let Some(val_prec) = logs.val_precision {
                self.val_precisions.push(val_prec);
            }
        }
    }
}

#[test]
fn loads_dataset_trains_and_generates_learning_curve_per_iteration() {
    let csv_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));
    let dataset = load_dataset(&csv_path, 1, build_dataset_column_names(), 0)
        .expect("loading data/data.csv should succeed");

    // Baseline architecture input uses all 30 numeric features and diagnosis as binary target.
    let x_raw = dataset.features.slice(s![.., 1..]).to_owned();
    let y = dataset
        .features
        .column(0)
        .mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });

    let n = x_raw.nrows();
    let train_end = (0.85 * n as f64).round() as usize;
    let val_end = n;

    let x_train_raw = x_raw.slice(s![0..train_end, ..]).to_owned();
    let x_val_raw = x_raw.slice(s![train_end..val_end, ..]).to_owned();
    let y_train = y.slice(s![0..train_end]).to_owned();
    let y_val = y.slice(s![train_end..val_end]).to_owned();

    let (x_train, x_val) = standardize_from_train(&x_train_raw, &x_val_raw);

    // Baseline notebook architecture: 30 -> 24 -> 24 -> 24 -> 2
    let mut network = Network::builder()
        .learning_rate(0.0314)
        .add_layer(Layer::new(
            30,
            24,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            24,
            24,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            24,
            24,
            ActivationFunction::Sigmoid,
            WeightInitializer::He,
        ))
        .add_layer(Layer::new(
            24,
            2,
            ActivationFunction::Softmax,
            WeightInitializer::He,
        ))
        .build();

    let iterations = 84;
    let mut callback = MetricsCallback::new();
    let mut progress_logger = ProgressLogger::new(iterations);
    let mut callbacks: Vec<&mut dyn Callback> = vec![&mut callback, &mut progress_logger];

    let metrics = network.fit_with_callbacks(
        x_train.view(),
        y_train.view(),
        Some((x_val.view(), y_val.view())),
        mlp::network::model::FitConfig {
            batch_size: 8,
            epochs: iterations,
            optimizer: OptimizerType::SGD,
            loss_fn: LossFunction::CategoricalCrossEntropy,
        },
        &mut callbacks,
    );

    drop(callbacks);

    assert_eq!(callback.train_losses.len(), iterations);
    assert_eq!(callback.val_losses.len(), iterations);
    assert_eq!(callback.train_accuracies.len(), iterations);
    assert_eq!(callback.val_accuracies.len(), iterations);
    assert!(callback.train_losses.iter().all(|v| v.is_finite()));
    assert!(callback.val_losses.iter().all(|v| v.is_finite()));
    assert!(callback.train_accuracies.iter().all(|v| v.is_finite()));
    assert!(callback.val_accuracies.iter().all(|v| v.is_finite()));
    assert!(metrics.train_loss.is_finite());
    assert!(metrics.val_loss.is_finite());
    assert!(metrics.train_accuracy.is_finite());
    assert!(metrics.val_accuracy.is_finite());

    if should_write_png_artifact() {
        let output_dir = format!("{}/target/test-artifacts", env!("CARGO_MANIFEST_DIR"));
        fs::create_dir_all(&output_dir).expect("test artifact directory should be creatable");

        let history = TrainingHistory {
            train_loss: callback.train_losses,
            val_loss: callback.val_losses,
            train_accuracy: callback.train_accuracies,
            val_accuracy: callback.val_accuracies,
            train_precision: callback.train_precisions,
            val_precision: callback.val_precisions,
        };

        let loss_path = format!("{}/learning_curve_per_iteration.png", output_dir);
        plot_loss_curve(&history, &loss_path).expect("learning-curve plot should be generated");
        maybe_open_plot(&loss_path);

        let acc_path = format!("{}/learning_curve_accuracy.png", output_dir);
        plot_accuracy_curve(&history, &acc_path).expect("accuracy curve should be generated");
        maybe_open_plot(&acc_path);

        let combined_path = format!("{}/learning_curves_combined.png", output_dir);
        plot_training_curves(&history, &combined_path, &[])
            .expect("combined curves should be generated");
        maybe_open_plot(&combined_path);

        for path in [&loss_path, &acc_path, &combined_path] {
            let metadata = fs::metadata(path).expect("learning-curve image should exist");
            assert!(metadata.len() > 0);
        }
    }
}
