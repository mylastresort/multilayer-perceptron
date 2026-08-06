use std::error::Error;

use crate::data::preprocessing::Normalizer;
use crate::network::model::Network;
use crate::training::loss::assert_binary_output;
use crate::training::metrics::{
    ClassificationReport, compute_classification_report, compute_classification_scores_from_labels,
};
use ndarray::Array2;

use super::training::{build_dataset, extract_features_target};

pub struct PredictArgs {
    pub dataset_path: String,
    pub model_path: String,
}

fn print_classification_report(report: &ClassificationReport) {
    println!("  Classification report:");
    println!(
        "  {:<6}{:>12}{:>12}{:>12}{:>9}",
        "class", "precision", "recall", "f1", "support"
    );
    for c in &report.classes {
        println!(
            "  {:<6}{:>11.2}%{:>11.2}%{:>11.2}%{:>9}",
            c.class_id,
            c.precision * 100.0,
            c.recall * 100.0,
            c.f1 * 100.0,
            c.support
        );
    }
}

pub fn run_predict(args: &PredictArgs) -> Result<(), Box<dyn Error>> {
    let dataset = build_dataset(&args.dataset_path)?;
    let mut network = Network::load(&args.model_path)?;
    assert_binary_output(network.output_size());

    let (x_raw, y) = extract_features_target(&dataset);
    if x_raw.nrows() == 0 {
        return Err("dataset has no rows".into());
    }
    let Some(scaler) = network.scaler.as_ref() else {
        return Err("model has no feature scaler; retrain or save it after training".into());
    };
    let x = scaler.transform(&x_raw);
    let predictions: Array2<f64> = network.predict(&x);

    let loss = network
        .loss
        .compute(predictions.view(), y.view())
        .mean()
        .unwrap_or(0.0);

    let n = predictions.nrows();
    let accuracy = compute_classification_scores_from_labels(predictions.view(), y.view()).accuracy;

    println!("Prediction results on {} samples", n);
    println!("  {} loss : {loss:.6}", network.loss.as_str());
    println!("  Accuracy                  : {:.2}%", accuracy * 100.0);

    let report = compute_classification_report(predictions.view(), y.view());
    print_classification_report(&report);

    Ok(())
}

