use std::fs;
use std::process::Command;
use std::time::Duration;

use minifb::{Window, WindowOptions};
use mlp::data::loader::load_dataset;
use mlp::network::activation::ActivationFunction;
use mlp::network::callbacks::{Callback, CallbackLogs, ProgressLogger};
use mlp::network::initializer::WeightInitializer;
use mlp::network::layer::Layer;
use mlp::network::model::Network;
use mlp::training::loss::LossFunction;
use mlp::training::optimizer::OptimizerType;
use mlp::visualization::plotter::{TrainingHistory, plot_loss_curve};
use ndarray::{Array2, Axis, s};
use plotters::coord::Shift;
use plotters::prelude::*;

const TRAINING_COLOR: RGBColor = RGBColor(31, 119, 180);
const VALIDATION_COLOR: RGBColor = RGBColor(255, 127, 14);
const PANEL_BG_COLOR: RGBColor = RGBColor(236, 238, 240);
const GRID_COLOR: RGBColor = RGBColor(190, 190, 190);
const WINDOW_BG_COLOR: RGBColor = RGBColor(245, 245, 245);

struct LiveLossWindow {
    window: Window,
    width: usize,
    height: usize,
    buffer: Vec<u32>,
    rgb_buffer: Vec<u8>,
    frame_delay: Duration,
}

impl LiveLossWindow {
    fn new(title: &str, width: usize, height: usize) -> Option<Self> {
        let options = WindowOptions {
            borderless: false,
            title: true,
            resize: true,
            none: false,
            ..WindowOptions::default()
        };
        let mut window = Window::new(title, width, height, options).ok()?;
        window.set_cursor_visibility(true);
        window.set_target_fps(60);

        let delay_ms = std::env::var("MLP_LIVE_PLOT_DELAY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(10);

        Some(Self {
            window,
            width,
            height,
            buffer: vec![0x00FFFFFF; width * height],
            rgb_buffer: vec![255; width * height * 3],
            frame_delay: Duration::from_millis(delay_ms),
        })
    }

    fn update(
        &mut self,
        train_losses: &[f64],
        val_losses: &[f64],
        train_accuracies: &[f64],
        val_accuracies: &[f64],
    ) {
        if !self.window.is_open() {
            return;
        }

        // Some Linux backends can reset cursor visibility on enter/focus events.
        // Re-apply cursor settings every frame to keep the pointer visible.
        self.window.set_cursor_visibility(true);

        self.render_chart(train_losses, val_losses, train_accuracies, val_accuracies);
        self.sync_rgb_to_u32();

        let _ = self
            .window
            .update_with_buffer(&self.buffer, self.width, self.height);
        std::thread::sleep(self.frame_delay);
    }

    fn render_chart(
        &mut self,
        train_losses: &[f64],
        val_losses: &[f64],
        train_accuracies: &[f64],
        val_accuracies: &[f64],
    ) {
        let drawing_area = BitMapBackend::with_buffer(
            &mut self.rgb_buffer,
            (self.width as u32, self.height as u32),
        )
        .into_drawing_area();

        let _ = drawing_area.fill(&WINDOW_BG_COLOR);

        if train_losses.is_empty() {
            let _ = drawing_area.present();
            return;
        }

        let areas = drawing_area.split_evenly((1, 2));
        let loss_area = &areas[0];
        let acc_area = &areas[1];

        Self::draw_loss_panel(loss_area, train_losses, val_losses);
        Self::draw_accuracy_panel(acc_area, train_losses, train_accuracies, val_accuracies);

        let _ = drawing_area.present();
    }

    fn finite_min_max(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for value in values {
            min_v = min_v.min(value);
            max_v = max_v.max(value);
        }

        if min_v.is_finite() && max_v.is_finite() {
            Some((min_v, max_v))
        } else {
            None
        }
    }

    fn padded_range(min_v: f64, max_v: f64, floor: Option<f64>, ceil: Option<f64>) -> (f64, f64) {
        let spread = (max_v - min_v).max(1e-9);
        let mut low = min_v - spread * 0.1;
        let mut high = max_v + spread * 0.1;

        if let Some(floor) = floor {
            low = low.max(floor);
        }
        if let Some(ceil) = ceil {
            high = high.min(ceil);
        }

        (low, high)
    }

    fn draw_loss_panel(
        area: &DrawingArea<BitMapBackend<'_>, Shift>,
        train_losses: &[f64],
        val_losses: &[f64],
    ) {
        let Some((min_loss, max_loss)) = Self::finite_min_max(
            train_losses
                .iter()
                .copied()
                .chain(val_losses.iter().copied()),
        ) else {
            return;
        };

        let (y_low, y_high) = Self::padded_range(min_loss, max_loss, None, None);
        if let Ok(mut chart) = ChartBuilder::on(area)
            .caption("Loss", ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(72)
            .y_label_area_size(95)
            .build_cartesian_2d(0..(train_losses.len() as i32), y_low..y_high)
        {
            let _ = chart.plotting_area().fill(&PANEL_BG_COLOR);
            let _ = chart
                .configure_mesh()
                .x_desc("epochs")
                .y_desc("loss")
                .axis_desc_style(("sans-serif", 22))
                .label_style(("sans-serif", 18))
                .light_line_style(GRID_COLOR.mix(0.7))
                .bold_line_style(GRID_COLOR.mix(0.3))
                .draw();

            let _ = chart
                .draw_series(LineSeries::new(
                    train_losses
                        .iter()
                        .enumerate()
                        .map(|(i, loss)| (i as i32, *loss)),
                    TRAINING_COLOR.stroke_width(3),
                ))
                .map(|series| {
                    series.label("training loss").legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 16, y)], TRAINING_COLOR.stroke_width(3))
                    });
                });

            if !val_losses.is_empty() {
                let _ = chart
                    .draw_series(LineSeries::new(
                        val_losses
                            .iter()
                            .enumerate()
                            .map(|(i, loss)| (i as i32, *loss)),
                        VALIDATION_COLOR.stroke_width(3),
                    ))
                    .map(|series| {
                        series.label("validation loss").legend(|(x, y)| {
                            PathElement::new(
                                vec![(x, y), (x + 16, y)],
                                VALIDATION_COLOR.stroke_width(3),
                            )
                        });
                    });
            }

            let _ = chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .label_font(("sans-serif", 20))
                .border_style(GRID_COLOR)
                .background_style(WHITE.mix(0.75))
                .draw();
        }
    }

    fn draw_accuracy_panel(
        area: &DrawingArea<BitMapBackend<'_>, Shift>,
        train_losses: &[f64],
        train_accuracies: &[f64],
        val_accuracies: &[f64],
    ) {
        let (min_acc, max_acc) = Self::finite_min_max(
            train_accuracies
                .iter()
                .copied()
                .chain(val_accuracies.iter().copied()),
        )
        .map(|(min_v, max_v)| Self::padded_range(min_v, max_v, Some(0.0), Some(1.0)))
        .unwrap_or((0.0, 1.0));

        if let Ok(mut acc_chart) = ChartBuilder::on(area)
            .caption("Learning Curves", ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(72)
            .y_label_area_size(95)
            .build_cartesian_2d(0..(train_losses.len() as i32), min_acc..max_acc)
        {
            let _ = acc_chart.plotting_area().fill(&PANEL_BG_COLOR);
            let _ = acc_chart
                .configure_mesh()
                .x_desc("Epochs")
                .y_desc("Accuracy")
                .axis_desc_style(("sans-serif", 22))
                .label_style(("sans-serif", 18))
                .light_line_style(GRID_COLOR.mix(0.7))
                .bold_line_style(GRID_COLOR.mix(0.3))
                .draw();

            if !train_accuracies.is_empty() {
                let _ = acc_chart
                    .draw_series(LineSeries::new(
                        train_accuracies
                            .iter()
                            .enumerate()
                            .map(|(i, acc)| (i as i32, *acc)),
                        TRAINING_COLOR.stroke_width(3),
                    ))
                    .map(|series| {
                        series.label("training acc").legend(|(x, y)| {
                            PathElement::new(
                                vec![(x, y), (x + 16, y)],
                                TRAINING_COLOR.stroke_width(3),
                            )
                        });
                    });
            }

            if !val_accuracies.is_empty() {
                let _ = acc_chart
                    .draw_series(LineSeries::new(
                        val_accuracies
                            .iter()
                            .enumerate()
                            .map(|(i, acc)| (i as i32, *acc)),
                        VALIDATION_COLOR.stroke_width(3),
                    ))
                    .map(|series| {
                        series.label("validation acc").legend(|(x, y)| {
                            PathElement::new(
                                vec![(x, y), (x + 16, y)],
                                VALIDATION_COLOR.stroke_width(3),
                            )
                        });
                    });
            }

            let _ = acc_chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperLeft)
                .label_font(("sans-serif", 20))
                .border_style(GRID_COLOR)
                .background_style(WHITE.mix(0.75))
                .draw();
        }
    }

    fn sync_rgb_to_u32(&mut self) {
        for (pixel_idx, rgb_idx) in
            (0..self.buffer.len()).zip((0..self.rgb_buffer.len()).step_by(3))
        {
            let r = self.rgb_buffer[rgb_idx] as u32;
            let g = self.rgb_buffer[rgb_idx + 1] as u32;
            let b = self.rgb_buffer[rgb_idx + 2] as u32;
            self.buffer[pixel_idx] = (r << 16) | (g << 8) | b;
        }
    }

    fn keep_open_until_closed(&mut self) {
        while self.window.is_open() {
            self.window.update();
            std::thread::sleep(Duration::from_millis(16));
        }
    }
}

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

fn maybe_open_live_window() -> Option<LiveLossWindow> {
    if std::env::var("MLP_LIVE_PLOT").as_deref() != Ok("1") {
        return None;
    }
    if std::env::var("CI").is_ok() {
        return None;
    }
    LiveLossWindow::new("MLP Live Training Loss", 1800, 1120)
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

struct LiveLossCallback {
    train_losses: Vec<f64>,
    val_losses: Vec<f64>,
    train_accuracies: Vec<f64>,
    val_accuracies: Vec<f64>,
    live_window: Option<LiveLossWindow>,
}

impl LiveLossCallback {
    fn new(live_window: Option<LiveLossWindow>) -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            live_window,
        }
    }

    fn apply_logs(&mut self, logs: &CallbackLogs) {
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
    }
}

impl Callback for LiveLossCallback {
    fn on_epoch_end(&mut self, _epoch: usize, logs: Option<&CallbackLogs>) {
        if let Some(logs) = logs {
            self.apply_logs(logs);

            if let Some(window) = self.live_window.as_mut() {
                window.update(
                    &self.train_losses,
                    &self.val_losses,
                    &self.train_accuracies,
                    &self.val_accuracies,
                );
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
    let train_end = (0.70 * n as f64).round() as usize;
    let val_end = (0.85 * n as f64).round() as usize;

    let x_train_raw = x_raw.slice(s![0..train_end, ..]).to_owned();
    let x_val_raw = x_raw.slice(s![train_end..val_end, ..]).to_owned();
    let y_train = y.slice(s![0..train_end]).to_owned();
    let y_val = y.slice(s![train_end..val_end]).to_owned();

    let (x_train, x_val) = standardize_from_train(&x_train_raw, &x_val_raw);

    // Baseline notebook architecture: 30 -> 24 -> 24 -> 24 -> 2
    let mut network = Network::new()
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
    let mut callback = LiveLossCallback::new(maybe_open_live_window());
    let mut progress_logger = ProgressLogger::new(iterations);
    let mut callbacks: Vec<&mut dyn Callback> = vec![&mut callback, &mut progress_logger];

    let metrics = network.fit_with_callbacks(
        x_train.view(),
        y_train.view(),
        Some((x_val.view(), y_val.view())),
        8,
        iterations,
        OptimizerType::SGD,
        LossFunction::CategoricalCrossEntropy,
        &mut callbacks,
    );

    drop(callbacks);

    if let Some(window) = callback.live_window.as_mut() {
        window.keep_open_until_closed();
    }

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

        let output_path = format!("{}/learning_curve_per_iteration.png", output_dir);
        let history = TrainingHistory {
            train_loss: callback.train_losses,
            val_loss: callback.val_losses,
            train_accuracy: callback.train_accuracies,
            val_accuracy: callback.val_accuracies,
        };
        plot_loss_curve(&history, &output_path).expect("learning-curve plot should be generated");
        maybe_open_plot(&output_path);

        let metadata = fs::metadata(&output_path).expect("learning-curve image should exist");
        assert!(metadata.len() > 0);
    }
}
