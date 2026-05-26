use std::time::Duration;

use minifb::{Key, KeyRepeat, Window, WindowOptions};
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::{
    network::callbacks::{Callback, CallbackLogs},
    training::monitor::MonitoredMetric,
};

const TRAINING_COLOR: RGBColor = RGBColor(31, 119, 180);
const VALIDATION_COLOR: RGBColor = RGBColor(255, 127, 14);
const PANEL_BG_COLOR: RGBColor = RGBColor(236, 238, 240);
const GRID_COLOR: RGBColor = RGBColor(190, 190, 190);
const WINDOW_BG_COLOR: RGBColor = RGBColor(245, 245, 245);

#[derive(Debug, Clone)]
pub struct GuiMonitorConfig {
    pub enabled: bool,
    pub width: usize,
    pub height: usize,
    pub delay_ms: u64,
    pub metrics: Vec<MonitoredMetric>,
}

impl GuiMonitorConfig {
    pub fn from_env(enabled: bool, metrics: Vec<MonitoredMetric>) -> Self {
        let width = std::env::var("MLP_LIVE_PLOT_WIDTH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1800);
        let height = std::env::var("MLP_LIVE_PLOT_HEIGHT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1120);
        let delay_ms = std::env::var("MLP_LIVE_PLOT_DELAY_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(10);

        Self {
            enabled,
            width,
            height,
            delay_ms,
            metrics,
        }
    }
}

#[derive(Clone)]
struct MetricSeries {
    metric: MonitoredMetric,
    train_values: Vec<f64>,
    val_values: Vec<f64>,
}

pub struct LiveTrainingMonitorCallback {
    series: Vec<MetricSeries>,
    live_window: Option<LiveMonitorWindow>,
}

impl LiveTrainingMonitorCallback {
    pub fn new(config: GuiMonitorConfig) -> Self {
        let series = config
            .metrics
            .iter()
            .copied()
            .map(|metric| MetricSeries {
                metric,
                train_values: Vec::new(),
                val_values: Vec::new(),
            })
            .collect();

        let live_window = maybe_open_live_window(
            config.enabled,
            config.width,
            config.height,
            config.delay_ms,
        );

        Self {
            series,
            live_window,
        }
    }

    pub fn keep_open_until_closed(&mut self) {
        if let Some(window) = self.live_window.as_mut() {
            window.keep_open_until_closed();
        }
    }

    pub fn history_len(&self) -> usize {
        self.series
            .first()
            .map(|item| item.train_values.len())
            .unwrap_or(0)
    }
}

impl Callback for LiveTrainingMonitorCallback {
    fn on_epoch_end(&mut self, _epoch: usize, logs: Option<&CallbackLogs>) {
        let Some(logs) = logs else {
            return;
        };

        for metric_series in &mut self.series {
            metric_series
                .train_values
                .push(metric_series.metric.train_value(logs).unwrap_or(f64::NAN));
            metric_series
                .val_values
                .push(metric_series.metric.val_value(logs).unwrap_or(f64::NAN));
        }

        let mut close_requested = false;
        if let Some(window) = self.live_window.as_mut() {
            if !window.update(&self.series) {
                close_requested = true;
            }
        }

        if close_requested {
            self.live_window = None;
        }
    }
}

struct LiveMonitorWindow {
    window: Window,
    width: usize,
    height: usize,
    buffer: Vec<u32>,
    rgb_buffer: Vec<u8>,
    frame_delay: Duration,
}

impl LiveMonitorWindow {
    fn new(title: &str, width: usize, height: usize, delay_ms: u64) -> Option<Self> {
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

        Some(Self {
            window,
            width,
            height,
            buffer: vec![0x00FFFFFF; width * height],
            rgb_buffer: vec![255; width * height * 3],
            frame_delay: Duration::from_millis(delay_ms),
        })
    }

    fn update(&mut self, metrics: &[MetricSeries]) -> bool {
        if !self.window.is_open() {
            return false;
        }

        self.window.set_cursor_visibility(true);
        self.render_chart(metrics);
        self.sync_rgb_to_u32();

        let _ = self
            .window
            .update_with_buffer(&self.buffer, self.width, self.height);

        // Keyboard state is reliable after update/update_with_buffer pumps events.
        if self.window.is_key_pressed(Key::Space, KeyRepeat::No) {
            return false;
        }

        std::thread::sleep(self.frame_delay);

        true
    }

    fn render_chart(&mut self, metrics: &[MetricSeries]) {
        let drawing_area = BitMapBackend::with_buffer(
            &mut self.rgb_buffer,
            (self.width as u32, self.height as u32),
        )
        .into_drawing_area();

        let _ = drawing_area.fill(&WINDOW_BG_COLOR);
        if metrics.is_empty() {
            let _ = drawing_area.present();
            return;
        }

        let panel_count = metrics.len();
        let cols = (panel_count as f64).sqrt().ceil() as usize;
        let rows = panel_count.div_ceil(cols);
        let areas = drawing_area.split_evenly((rows, cols));

        for (metric_series, area) in metrics.iter().zip(areas.iter()) {
            Self::draw_metric_panel(area, metric_series);
        }

        let _ = drawing_area.present();
    }

    fn finite_min_max(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for value in values.filter(|v| v.is_finite()) {
            min_v = min_v.min(value);
            max_v = max_v.max(value);
        }

        if min_v.is_finite() && max_v.is_finite() {
            Some((min_v, max_v))
        } else {
            None
        }
    }

    fn draw_metric_panel(area: &DrawingArea<BitMapBackend<'_>, Shift>, metric_series: &MetricSeries) {
        let points = metric_series.train_values.len();
        if points == 0 {
            return;
        }

        let title = metric_series.metric.as_str();

        let values = metric_series
            .train_values
            .iter()
            .copied()
            .chain(metric_series.val_values.iter().copied());

        let (min_y, max_y) = Self::finite_min_max(values)
            .map(|(min_v, max_v)| {
                if matches!(
                    metric_series.metric,
                    MonitoredMetric::Accuracy
                        | MonitoredMetric::Precision
                        | MonitoredMetric::Recall
                        | MonitoredMetric::F1
                ) {
                    let spread = (max_v - min_v).max(1e-9);
                    ((min_v - 0.1 * spread).max(0.0), (max_v + 0.1 * spread).min(1.0))
                } else {
                    let spread = (max_v - min_v).max(1e-9);
                    (min_v - 0.1 * spread, max_v + 0.1 * spread)
                }
            })
            .unwrap_or((0.0, 1.0));

        if let Ok(mut chart) = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 22))
            .margin(15)
            .x_label_area_size(55)
            .y_label_area_size(70)
            .build_cartesian_2d(0..(points as i32), min_y..max_y)
        {
            let _ = chart.plotting_area().fill(&PANEL_BG_COLOR);
            let _ = chart
                .configure_mesh()
                .x_desc("epochs")
                .y_desc(metric_series.metric.as_str())
                .axis_desc_style(("sans-serif", 16))
                .label_style(("sans-serif", 14))
                .light_line_style(GRID_COLOR.mix(0.7))
                .bold_line_style(GRID_COLOR.mix(0.3))
                .draw();

            let _ = chart
                .draw_series(LineSeries::new(
                    metric_series
                        .train_values
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.is_finite())
                        .map(|(i, v)| (i as i32, *v)),
                    TRAINING_COLOR.stroke_width(3),
                ))
                .map(|series| {
                    series.label("train").legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 16, y)], TRAINING_COLOR.stroke_width(3))
                    });
                });

            let _ = chart
                .draw_series(LineSeries::new(
                    metric_series
                        .val_values
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| v.is_finite())
                        .map(|(i, v)| (i as i32, *v)),
                    VALIDATION_COLOR.stroke_width(3),
                ))
                .map(|series| {
                    series.label("val").legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 16, y)], VALIDATION_COLOR.stroke_width(3))
                    });
                });

            let _ = chart
                .configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .label_font(("sans-serif", 14))
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

            // Read key events after update() has processed window input.
            if self.window.is_key_pressed(Key::Space, KeyRepeat::No) {
                break;
            }

            std::thread::sleep(Duration::from_millis(16));
        }
    }
}

fn maybe_open_live_window(
    enabled: bool,
    width: usize,
    height: usize,
    delay_ms: u64,
) -> Option<LiveMonitorWindow> {
    if !enabled {
        return None;
    }

    if std::env::var("CI").is_ok() {
        return None;
    }

    LiveMonitorWindow::new("MLP Live Training Monitor", width, height, delay_ms)
}
