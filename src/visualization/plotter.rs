use plotters::coord::Shift;
use plotters::coord::types::{RangedCoordf64, RangedCoordi32};
use plotters::prelude::*;

use crate::training::monitor::MonitoredMetric;

pub struct TrainingHistory {
    pub train_loss: Vec<f64>,
    pub val_loss: Vec<f64>,
    pub train_accuracy: Vec<f64>,
    pub val_accuracy: Vec<f64>,
    pub train_precision: Vec<f64>,
    pub val_precision: Vec<f64>,
}

type PlotResult = Result<(), Box<dyn std::error::Error>>;

const PANEL_H: u32 = 360;
const PANEL_W: u32 = 700;
const PANEL_COLS: usize = 2;

fn draw_single_curve(
    root: &DrawingArea<BitMapBackend, Shift>,
    caption: &str,
    y_desc: &str,
    y_low: f64,
    y_high: f64,
    train: &[f64],
    val: &[f64],
) -> PlotResult {
    let mut chart = ChartBuilder::on(root)
        .caption(caption, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..(train.len() as i32), y_low..y_high)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc(y_desc)
        .draw()?;

    chart.draw_series(LineSeries::new(
        train.iter().enumerate().map(|(i, v)| (i as i32, *v)),
        &RED,
    ))?;

    if !val.is_empty() {
        chart.draw_series(LineSeries::new(
            val.iter().enumerate().map(|(i, v)| (i as i32, *v)),
            &BLUE,
        ))?;
    }

    Ok(())
}

pub fn plot_loss_curve(history: &TrainingHistory, output_path: &str) -> PlotResult {
    if history.train_loss.is_empty() {
        return Err("train_loss cannot be empty".into());
    }
    let (y_low, y_high) = domain_around(
        history
            .train_loss
            .iter()
            .copied()
            .chain(history.val_loss.iter().copied()),
    )?;

    let root = BitMapBackend::new(output_path, (900, 560)).into_drawing_area();
    root.fill(&WHITE)?;
    draw_single_curve(
        &root,
        "Learning Curve per Iteration",
        "Binary Cross-Entropy Loss",
        y_low,
        y_high,
        &history.train_loss,
        &history.val_loss,
    )?;
    root.present()?;
    Ok(())
}

pub fn plot_accuracy_curve(history: &TrainingHistory, output_path: &str) -> PlotResult {
    if history.train_accuracy.is_empty() {
        return Err("train_accuracy cannot be empty".into());
    }
    let (low, high) = domain_around(
        history
            .train_accuracy
            .iter()
            .copied()
            .chain(history.val_accuracy.iter().copied()),
    )?;

    let root = BitMapBackend::new(output_path, (900, 560)).into_drawing_area();
    root.fill(&WHITE)?;
    draw_single_curve(
        &root,
        "Accuracy per Iteration",
        "Accuracy",
        low.max(0.0),
        high.min(1.0),
        &history.train_accuracy,
        &history.val_accuracy,
    )?;
    root.present()?;
    Ok(())
}

type MetricChart<'a, 'b> =
    ChartContext<'a, BitMapBackend<'b>, Cartesian2d<RangedCoordi32, RangedCoordf64>>;

fn draw_metric_series(
    chart: &mut MetricChart<'_, '_>,
    values: &[f64],
    color: RGBColor,
    label: &str,
) -> PlotResult {
    if values.is_empty() {
        return Ok(());
    }
    chart
        .draw_series(LineSeries::new(
            values.iter().enumerate().map(|(i, v)| (i as i32, *v)),
            color,
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 16, y)], color));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_metric_panel(
    area: &DrawingArea<BitMapBackend, Shift>,
    caption: &str,
    y_desc: &str,
    y_low: f64,
    y_high: f64,
    n: i32,
    train: &[f64],
    val: &[f64],
    train_label: &str,
    val_label: &str,
    legend_pos: SeriesLabelPosition,
) -> PlotResult {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 24))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(55)
        .build_cartesian_2d(0..n, y_low..y_high)?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc(y_desc)
        .draw()?;

    draw_metric_series(&mut chart, train, RED, train_label)?;
    draw_metric_series(&mut chart, val, BLUE, val_label)?;

    chart
        .configure_series_labels()
        .position(legend_pos)
        .border_style(BLACK)
        .background_style(WHITE)
        .draw()?;

    Ok(())
}

fn domain_around(values: impl Iterator<Item = f64>) -> Result<(f64, f64), String> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in values {
        min = min.min(v);
        max = max.max(v);
    }
    if !min.is_finite() || !max.is_finite() {
        return Err("metric values must be finite".into());
    }
    let spread = (max - min).max(1e-6);
    Ok((min - spread * 0.1, max + spread * 0.1))
}

fn selected_metrics(metrics: &[MonitoredMetric]) -> Vec<MonitoredMetric> {
    if metrics.is_empty() {
        vec![
            MonitoredMetric::Loss,
            MonitoredMetric::Accuracy,
            MonitoredMetric::Precision,
        ]
    } else {
        metrics.to_vec()
    }
}

struct MetricDomains {
    loss: (f64, f64),
    accuracy: (f64, f64),
    precision: (f64, f64),
}

fn metric_domains(history: &TrainingHistory) -> Result<MetricDomains, String> {
    let loss = domain_around(
        history
            .train_loss
            .iter()
            .copied()
            .chain(history.val_loss.iter().copied()),
    )?;
    let (acc_low, acc_high) = domain_around(
        history
            .train_accuracy
            .iter()
            .copied()
            .chain(history.val_accuracy.iter().copied()),
    )?;
    let (prec_low, prec_high) = domain_around(
        history
            .train_precision
            .iter()
            .copied()
            .chain(history.val_precision.iter().copied()),
    )?;
    Ok(MetricDomains {
        loss,
        accuracy: (acc_low.max(0.0), acc_high.min(1.0)),
        precision: (prec_low.max(0.0), prec_high.min(1.0)),
    })
}

struct PanelSpec<'a> {
    caption: &'a str,
    y_desc: &'a str,
    domain: (f64, f64),
    train: &'a [f64],
    val: &'a [f64],
    train_label: &'a str,
    val_label: &'a str,
    legend_pos: SeriesLabelPosition,
}

struct MetricSeries<'a> {
    train: &'a [f64],
    val: &'a [f64],
}

fn metric_series<'a>(history: &'a TrainingHistory, metric: MonitoredMetric) -> MetricSeries<'a> {
    match metric {
        MonitoredMetric::Loss => MetricSeries {
            train: &history.train_loss,
            val: &history.val_loss,
        },
        MonitoredMetric::Accuracy => MetricSeries {
            train: &history.train_accuracy,
            val: &history.val_accuracy,
        },
        MonitoredMetric::Precision => MetricSeries {
            train: &history.train_precision,
            val: &history.val_precision,
        },
    }
}

fn metric_domain(domains: &MetricDomains, metric: MonitoredMetric) -> (f64, f64) {
    match metric {
        MonitoredMetric::Loss => domains.loss,
        MonitoredMetric::Accuracy => domains.accuracy,
        MonitoredMetric::Precision => domains.precision,
    }
}

fn metric_meta(
    metric: MonitoredMetric,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    SeriesLabelPosition,
) {
    match metric {
        MonitoredMetric::Loss => (
            "Loss",
            "Loss",
            "training loss",
            "validation loss",
            SeriesLabelPosition::UpperRight,
        ),
        MonitoredMetric::Accuracy => (
            "Accuracy",
            "Accuracy",
            "training accuracy",
            "validation accuracy",
            SeriesLabelPosition::LowerRight,
        ),
        MonitoredMetric::Precision => (
            "Precision",
            "Precision",
            "training precision",
            "validation precision",
            SeriesLabelPosition::LowerRight,
        ),
    }
}

fn panel_spec<'a>(
    history: &'a TrainingHistory,
    domains: &MetricDomains,
    metric: MonitoredMetric,
) -> PanelSpec<'a> {
    let series = metric_series(history, metric);
    let (caption, y_desc, train_label, val_label, legend_pos) = metric_meta(metric);
    PanelSpec {
        caption,
        y_desc,
        domain: metric_domain(domains, metric),
        train: series.train,
        val: series.val,
        train_label,
        val_label,
        legend_pos,
    }
}

fn draw_panel_grid(
    root: &DrawingArea<BitMapBackend, Shift>,
    panels: &[(PanelSpec<'_>, usize)],
) -> PlotResult {
    let rows = panels.len().div_ceil(PANEL_COLS);
    let mut vertical_rest = root.clone();
    for row in 0..rows {
        let (row_area, vertical_remainder) = vertical_rest.split_vertically(PANEL_H);
        let mut horizontal_rest = row_area.clone();
        for col in 0..PANEL_COLS {
            let index = row * PANEL_COLS + col;
            let Some((spec, n)) = panels.get(index) else {
                break;
            };
            let (panel, horizontal_remainder) = horizontal_rest.split_horizontally(PANEL_W);
            draw_metric_panel(
                &panel,
                spec.caption,
                spec.y_desc,
                spec.domain.0,
                spec.domain.1,
                *n as i32,
                spec.train,
                spec.val,
                spec.train_label,
                spec.val_label,
                spec.legend_pos.clone(),
            )?;
            horizontal_rest = horizontal_remainder;
        }
        vertical_rest = vertical_remainder;
    }
    Ok(())
}

pub fn plot_training_curves(
    history: &TrainingHistory,
    output_path: &str,
    metrics: &[MonitoredMetric],
) -> PlotResult {
    if history.train_loss.is_empty() {
        return Err("train_loss cannot be empty".into());
    }

    let selected = selected_metrics(metrics);
    let domains = metric_domains(history)?;
    let n = history.train_loss.len();
    let panels: Vec<(PanelSpec<'_>, usize)> = selected
        .iter()
        .map(|metric| (panel_spec(history, &domains, *metric), n))
        .collect();

    let rows = panels.len().div_ceil(PANEL_COLS);
    let root = BitMapBackend::new(
        output_path,
        (PANEL_W * PANEL_COLS as u32, PANEL_H * rows as u32),
    )
    .into_drawing_area();
    root.fill(&WHITE)?;

    draw_panel_grid(&root, &panels)?;
    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::training::monitor::MonitoredMetric;

    use super::{TrainingHistory, plot_accuracy_curve, plot_loss_curve, plot_training_curves};

    fn history_with_metrics() -> TrainingHistory {
        TrainingHistory {
            train_loss: vec![0.9, 0.5, 0.2],
            val_loss: vec![1.0, 0.6, 0.3],
            train_accuracy: vec![0.5, 0.8, 0.95],
            val_accuracy: vec![0.45, 0.75, 0.9],
            train_precision: vec![0.4, 0.7, 0.9],
            val_precision: vec![0.35, 0.65, 0.85],
        }
    }

    fn empty_history() -> TrainingHistory {
        TrainingHistory {
            train_loss: vec![],
            val_loss: vec![],
            train_accuracy: vec![],
            val_accuracy: vec![],
            train_precision: vec![],
            val_precision: vec![],
        }
    }

    #[test]
    fn plot_loss_curve_returns_error_when_train_loss_is_empty() {
        let history = empty_history();
        let result = plot_loss_curve(&history, "/tmp/mlp_test_plot_empty.png");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn plot_loss_curve_returns_error_when_loss_is_non_finite() {
        let mut history = empty_history();
        history.train_loss = vec![f64::INFINITY];
        let result = plot_loss_curve(&history, "/tmp/mlp_test_plot_inf.png");
        assert!(result.is_err());
    }

    #[test]
    fn plot_accuracy_curve_returns_error_when_train_accuracy_is_empty() {
        let mut history = empty_history();
        history.train_loss = vec![0.5];
        let result = plot_accuracy_curve(&history, "/tmp/mlp_test_plot_acc_empty.png");
        assert!(result.is_err());
    }

    #[test]
    fn plot_accuracy_curve_returns_error_when_accuracy_is_non_finite() {
        let mut history = empty_history();
        history.train_loss = vec![0.5];
        history.train_accuracy = vec![f64::NAN];
        let result = plot_accuracy_curve(&history, "/tmp/mlp_test_plot_acc_inf.png");
        assert!(result.is_err());
    }

    #[test]
    fn plot_loss_and_accuracy_curves_save_files() {
        let history = history_with_metrics();
        let loss_path = format!("/tmp/mlp_test_plot_ok_loss_{}.png", std::process::id());
        let acc_path = format!("/tmp/mlp_test_plot_ok_acc_{}.png", std::process::id());
        plot_loss_curve(&history, &loss_path).expect("loss curve should save");
        plot_accuracy_curve(&history, &acc_path).expect("accuracy curve should save");
        assert!(std::path::Path::new(&loss_path).exists());
        assert!(std::path::Path::new(&acc_path).exists());
        let _ = std::fs::remove_file(&loss_path);
        let _ = std::fs::remove_file(&acc_path);
    }

    #[test]
    fn plot_training_curves_returns_error_when_loss_is_empty() {
        let mut history = empty_history();
        history.train_accuracy = vec![0.5];
        let result = plot_training_curves(&history, "/tmp/mlp_test_curves_empty.png", &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn plot_training_curves_returns_error_when_loss_is_non_finite() {
        let mut history = empty_history();
        history.train_loss = vec![f64::NAN];
        history.train_accuracy = vec![0.5];
        let result = plot_training_curves(&history, "/tmp/mlp_test_curves_inf.png", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn plot_training_curves_saves_file() {
        let history = history_with_metrics();
        let path = format!("/tmp/mlp_test_plot_ok_curves_{}.png", std::process::id());
        plot_training_curves(&history, &path, &[]).expect("combined curves should save");
        assert!(std::path::Path::new(&path).exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn plot_training_curves_honors_metric_selection() {
        let history = history_with_metrics();
        let path = format!(
            "/tmp/mlp_test_plot_ok_curves_prec_{}.png",
            std::process::id()
        );
        plot_training_curves(&history, &path, &[MonitoredMetric::Precision])
            .expect("precision-only curves should save");
        assert!(std::path::Path::new(&path).exists());
        let _ = std::fs::remove_file(&path);
    }
}
