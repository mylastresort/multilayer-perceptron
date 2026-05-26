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

#[cfg(test)]
mod tests {
    use super::{run_split, SplitArgs};

    #[test]
    fn run_split_rejects_ratio_of_zero() {
        let result = run_split(&SplitArgs {
            dataset_path: format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR")),
            train_out: "/tmp/mlp_split_train_dummy.csv".to_string(),
            val_out: "/tmp/mlp_split_val_dummy.csv".to_string(),
            ratio: 0.0,
        });
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("(0, 1)"), "unexpected error: {msg}");
    }

    #[test]
    fn run_split_rejects_ratio_of_one() {
        let result = run_split(&SplitArgs {
            dataset_path: format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR")),
            train_out: "/tmp/mlp_split_train_dummy.csv".to_string(),
            val_out: "/tmp/mlp_split_val_dummy.csv".to_string(),
            ratio: 1.0,
        });
        assert!(result.is_err());
    }

    #[test]
    fn run_split_creates_train_and_val_csv_files() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let train_path = format!("/tmp/mlp_split_train_{}_{}.csv", std::process::id(), ts);
        let val_path = format!("/tmp/mlp_split_val_{}_{}.csv", std::process::id(), ts);

        let result = run_split(&SplitArgs {
            dataset_path: format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR")),
            train_out: train_path.clone(),
            val_out: val_path.clone(),
            ratio: 0.8,
        });

        let _ = std::fs::remove_file(&train_path);
        let _ = std::fs::remove_file(&val_path);

        assert!(result.is_ok(), "run_split failed: {:?}", result.err());
    }

    #[test]
    fn run_split_rejects_dataset_with_fewer_than_two_rows() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let csv_path = format!("/tmp/mlp_tiny_{}_{}.csv", std::process::id(), ts);
        // A CSV with exactly 1 data row (+ header) — n < 2 triggers line 46.
        std::fs::write(&csv_path, "id,diagnosis,f1\n1,M,0.5\n").unwrap();
        let result = run_split(&SplitArgs {
            dataset_path: csv_path.clone(),
            train_out: format!("/tmp/mlp_split_train_tiny_{ts}.csv"),
            val_out: format!("/tmp/mlp_split_val_tiny_{ts}.csv"),
            ratio: 0.7,
        });
        let _ = std::fs::remove_file(&csv_path);
        let Err(e) = result else { panic!("expected error for < 2 rows") };
        assert!(e.to_string().contains("at least 2 rows"), "unexpected: {e}");
    }
}
