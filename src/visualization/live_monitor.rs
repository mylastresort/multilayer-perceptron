use std::time::Duration;

use minifb::{Window, WindowOptions};
use plotters::coord::Shift;
use plotters::prelude::*;

use crate::network::callbacks::{Callback, CallbackLogs};

const TRAINING_COLOR: RGBColor = RGBColor(31, 119, 180);
const VALIDATION_COLOR: RGBColor = RGBColor(255, 127, 14);
const PANEL_BG_COLOR: RGBColor = RGBColor(236, 238, 240);
const GRID_COLOR: RGBColor = RGBColor(190, 190, 190);
const WINDOW_BG_COLOR: RGBColor = RGBColor(245, 245, 245);

pub struct LiveTrainingMonitorCallback {
    train_losses: Vec<f64>,
    val_losses: Vec<f64>,
    train_accuracies: Vec<f64>,
    val_accuracies: Vec<f64>,
    live_window: Option<LiveMonitorWindow>,
}

impl LiveTrainingMonitorCallback {
    pub fn from_env() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            live_window: maybe_open_live_window(),
        }
    }

    pub fn keep_open_until_closed(&mut self) {
        if let Some(window) = self.live_window.as_mut() {
            window.keep_open_until_closed();
        }
    }

    pub fn history_len(&self) -> usize {
        self.train_losses.len()
    }
}

impl Callback for LiveTrainingMonitorCallback {
    fn on_epoch_end(&mut self, _epoch: usize, logs: Option<&CallbackLogs>) {
        let Some(logs) = logs else {
            return;
        };

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

struct LiveMonitorWindow {
    window: Window,
    width: usize,
    height: usize,
    buffer: Vec<u32>,
    rgb_buffer: Vec<u8>,
    frame_delay: Duration,
}

impl LiveMonitorWindow {
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

        if let Ok(mut chart) = ChartBuilder::on(area)
            .caption("Accuracy", ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(72)
            .y_label_area_size(95)
            .build_cartesian_2d(0..(train_losses.len() as i32), min_acc..max_acc)
        {
            let _ = chart.plotting_area().fill(&PANEL_BG_COLOR);
            let _ = chart
                .configure_mesh()
                .x_desc("epochs")
                .y_desc("accuracy")
                .axis_desc_style(("sans-serif", 22))
                .label_style(("sans-serif", 18))
                .light_line_style(GRID_COLOR.mix(0.7))
                .bold_line_style(GRID_COLOR.mix(0.3))
                .draw();

            if !train_accuracies.is_empty() {
                let _ = chart
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
                let _ = chart
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

            let _ = chart
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

fn maybe_open_live_window() -> Option<LiveMonitorWindow> {
    if std::env::var("MLP_LIVE_PLOT").as_deref() != Ok("1") {
        return None;
    }

    if std::env::var("CI").is_ok() {
        return None;
    }

    let width = std::env::var("MLP_LIVE_PLOT_WIDTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1800);
    let height = std::env::var("MLP_LIVE_PLOT_HEIGHT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1120);

    LiveMonitorWindow::new("MLP Live Training Monitor", width, height)
}
