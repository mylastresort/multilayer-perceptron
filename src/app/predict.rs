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

    // Standardise using the training statistics stored in the model.
    let x = if let (Some(mean), Some(std)) = (&network.feature_mean, &network.feature_std) {
        (&x_raw - mean) / std
    } else {
        let means = x_raw
            .mean_axis(Axis(0))
            .expect("features should not be empty");
        let stds = x_raw.std_axis(Axis(0), 0.0).mapv(|v| v.max(1e-12));
        (&x_raw - &means) / &stds
    };

    let predictions: Array2<f64> = network.predict(&x);

    // Evaluate with binary cross-entropy (mean over the samples).
    let loss = LossFunction::BinaryCrossEntropy
        .compute(predictions.view(), y.view())
        .mean()
        .expect("prediction dataset has at least one row");

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

#[cfg(test)]
mod tests {
    use super::{PredictArgs, run_predict};
    use mlp::network::{
        activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
        model::Network,
    };

    #[test]
    fn run_predict_returns_error_for_missing_model() {
        let dataset_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));
        let result = run_predict(&PredictArgs {
            dataset_path,
            model_path: "/tmp/mlp_nonexistent_model_xyz123.json".to_string(),
        });
        assert!(result.is_err());
    }

    #[test]
    fn run_predict_returns_error_for_empty_dataset() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let csv_path = format!("/tmp/mlp_empty_{}_{}.csv", std::process::id(), ts);
        // Only a header row — 0 data rows → triggers "prediction dataset has no rows".
        std::fs::write(&csv_path, "id,diagnosis,f1\n").unwrap();

        // Build a minimal 1-output network compatible with 1 feature column.
        let model_path = format!("/tmp/mlp_pred_empty_{}_{}.json", std::process::id(), ts);
        let network = Network::new()
            .learning_rate(0.01)
            .add_layer(Layer::new(
                1,
                2,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                2,
                2,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                2,
                1,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .build();
        network.save(&model_path).expect("model should save");

        let result = run_predict(&PredictArgs {
            dataset_path: csv_path.clone(),
            model_path: model_path.clone(),
        });
        let _ = std::fs::remove_file(&csv_path);
        let _ = std::fs::remove_file(&model_path);
        // Should error because dataset has 0 rows.
        assert!(result.is_err());
    }

    #[test]
    fn run_predict_succeeds_with_single_output_network() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_path = format!("/tmp/mlp_predict_single_{}_{}.json", std::process::id(), ts);
        let dataset_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));

        // Single-output Sigmoid network: predictions.ncols() == 1 → exercises line 46.
        let network = Network::new()
            .learning_rate(0.01)
            .add_layer(Layer::new(
                30,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                1,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .build();
        network.save(&model_path).expect("model should save");

        let result = run_predict(&PredictArgs {
            dataset_path,
            model_path: model_path.clone(),
        });
        let _ = std::fs::remove_file(&model_path);
        assert!(result.is_ok(), "run_predict failed: {:?}", result.err());
    }

    #[test]
    fn run_predict_succeeds_with_two_output_network() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model_path = format!("/tmp/mlp_predict_two_{}_{}.json", std::process::id(), ts);
        let dataset_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));

        // Two-output Softmax network: predictions.ncols() == 2 → exercises the else-branch
        // (lines 48-55) where accuracy is computed via argmax.
        let network = Network::new()
            .learning_rate(0.01)
            .add_layer(Layer::new(
                30,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                2,
                ActivationFunction::Softmax,
                WeightInitializer::He,
            ))
            .build();
        network.save(&model_path).expect("model should save");

        let result = run_predict(&PredictArgs {
            dataset_path,
            model_path: model_path.clone(),
        });
        let _ = std::fs::remove_file(&model_path);
        assert!(result.is_ok(), "run_predict failed: {:?}", result.err());
    }
}
