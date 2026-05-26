use std::error::Error;
use std::io::Write;

use mlp::data::loader::Dataset;

use super::training::build_dataset;

pub struct SplitArgs {
    pub dataset_path: String,
    pub train_out: String,
    pub val_out: String,
    /// Fraction of rows assigned to the training split (0 < ratio < 1).
    pub ratio: f64,
}

/// Write a `Dataset` slice as CSV (feature columns followed by the label column).
fn write_csv(dataset: &Dataset, path: &str) -> Result<(), Box<dyn Error>> {
    let mut file = std::fs::File::create(path)?;

    // Header: feature names + "label"
    let header: Vec<&str> = dataset.feature_names.iter().map(String::as_str).collect();
    writeln!(file, "{},label", header.join(","))?;

    for row in 0..dataset.features.nrows() {
        let feature_cols: Vec<String> = (0..dataset.features.ncols())
            .map(|c| dataset.features[[row, c]].to_string())
            .collect();
        writeln!(file, "{},{}", feature_cols.join(","), dataset.labels[row])?;
    }

    Ok(())
}

pub fn run_split(args: &SplitArgs) -> Result<(), Box<dyn Error>> {
    if args.ratio <= 0.0 || args.ratio >= 1.0 {
        return Err(format!(
            "--ratio must be in (0, 1), got {}",
            args.ratio
        )
        .into());
    }

    let dataset = build_dataset(&args.dataset_path)?;
    let n = dataset.features.nrows();
    if n < 2 {
        return Err("dataset must have at least 2 rows to split".into());
    }

    let train_end = ((args.ratio * n as f64).round() as usize).max(1).min(n - 1);

    use ndarray::{Axis, s};
    let train = Dataset {
        features: dataset.features.slice(s![0..train_end, ..]).to_owned(),
        labels: dataset.labels.slice(s![0..train_end]).to_owned(),
        feature_names: dataset.feature_names.clone(),
    };
    let val = Dataset {
        features: dataset.features.slice(s![train_end.., ..]).to_owned(),
        labels: dataset.labels.slice(s![train_end..]).to_owned(),
        feature_names: dataset.feature_names.clone(),
    };

    write_csv(&train, &args.train_out)?;
    write_csv(&val, &args.val_out)?;

    println!(
        "Split complete: {} training rows → {}  |  {} validation rows → {}",
        train.features.nrows(),
        args.train_out,
        val.features.nrows(),
        args.val_out,
    );
    Ok(())
}
