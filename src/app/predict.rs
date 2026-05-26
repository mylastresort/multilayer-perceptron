use std::error::Error;

use mlp::network::model::Network;
use mlp::training::loss::{Loss, LossFunction};
use ndarray::{Array2, Axis, s};

use super::training::build_dataset;

pub struct PredictArgs {
    pub dataset_path: String,
    pub model_path: String,
}

pub fn run_predict(args: &PredictArgs) -> Result<(), Box<dyn Error>> {
    let dataset = build_dataset(&args.dataset_path)?;
    let mut network = Network::load(&args.model_path)?;

    // Prepare features and labels the same way training does.
    let x_raw = dataset.features.slice(s![.., 1..]).to_owned();
    let y = dataset
        .features
        .column(0)
        .mapv(|v| if v >= 0.5 { 1.0 } else { 0.0 });

    if x_raw.nrows() == 0 {
        return Err("prediction dataset has no rows".into());
    }

    // Standardise using the dataset's own mean/std (inference-only; no training split needed).
    let means = x_raw
        .mean_axis(Axis(0))
        .expect("features should not be empty");
    let stds = x_raw.std_axis(Axis(0), 0.0).mapv(|v| v.max(1e-12));
    let x = (x_raw - &means) / &stds;

    let predictions: Array2<f64> = network.predict(&x);

    // Evaluate with binary cross-entropy.
    let loss = LossFunction::BinaryCrossEntropy.compute(predictions.view(), y.view());

    // Compute accuracy: pick the column with the higher predicted probability.
    let n = predictions.nrows();
    let correct = (0..n)
        .filter(|&i| {
            let pred_class = if predictions.ncols() == 1 {
                if predictions[[i, 0]] >= 0.5 { 1.0 } else { 0.0 }
            } else {
                let row = predictions.row(i);
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                max_idx as f64
            };
            (pred_class - y[i]).abs() < 0.5
        })
        .count();

    let accuracy = correct as f64 / n as f64;

    println!("Prediction results on {} samples", n);
    println!("  Binary cross-entropy loss : {loss:.6}");
    println!("  Accuracy                  : {:.2}%", accuracy * 100.0);

    Ok(())
}
