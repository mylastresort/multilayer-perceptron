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
