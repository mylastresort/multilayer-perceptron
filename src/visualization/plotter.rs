use plotters::prelude::*;

pub struct TrainingHistory {
    pub train_loss: Vec<f64>,
    pub val_loss: Vec<f64>,
    pub train_accuracy: Vec<f64>,
    pub val_accuracy: Vec<f64>,
}

pub fn plot_loss_curve(
    history: &TrainingHistory,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if history.train_loss.is_empty() {
        return Err("train_loss cannot be empty".into());
    }

    let root = BitMapBackend::new(output_path, (900, 560)).into_drawing_area();
    root.fill(&WHITE)?;

    let loss_iter = history
        .train_loss
        .iter()
        .copied()
        .chain(history.val_loss.iter().copied());

    let mut min_loss = f64::INFINITY;
    let mut max_loss = f64::NEG_INFINITY;
    for loss in loss_iter {
        min_loss = min_loss.min(loss);
        max_loss = max_loss.max(loss);
    }

    if !min_loss.is_finite() || !max_loss.is_finite() {
        return Err("loss values must be finite".into());
    }

    let spread = (max_loss - min_loss).max(1e-6);
    let y_low = min_loss - (spread * 0.1);
    let y_high = max_loss + (spread * 0.1);

    let mut chart = ChartBuilder::on(&root)
        .caption("Learning Curve per Iteration", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..(history.train_loss.len() as i32), y_low..y_high)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Binary Cross-Entropy Loss")
        .draw()?;

    chart.draw_series(LineSeries::new(
        history
            .train_loss
            .iter()
            .enumerate()
            .map(|(i, loss)| (i as i32, *loss)),
        &RED,
    ))?;

    if !history.val_loss.is_empty() {
        chart.draw_series(LineSeries::new(
            history
                .val_loss
                .iter()
                .enumerate()
                .map(|(i, loss)| (i as i32, *loss)),
            &BLUE,
        ))?;
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{TrainingHistory, plot_loss_curve};

    #[test]
    fn plot_loss_curve_returns_error_when_train_loss_is_empty() {
        let history = TrainingHistory {
            train_loss: vec![],
            val_loss: vec![],
            train_accuracy: vec![],
            val_accuracy: vec![],
        };
        let result = plot_loss_curve(&history, "/tmp/mlp_test_plot_empty.png");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn plot_loss_curve_returns_error_when_loss_is_non_finite() {
        let history = TrainingHistory {
            train_loss: vec![f64::INFINITY],
            val_loss: vec![],
            train_accuracy: vec![],
            val_accuracy: vec![],
        };
        let result = plot_loss_curve(&history, "/tmp/mlp_test_plot_inf.png");
        assert!(result.is_err());
    }
}
