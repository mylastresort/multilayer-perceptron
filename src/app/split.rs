use std::error::Error;
use std::io::Write;

use crate::data::loader::Dataset;
use crate::data::split::stratified_split_by_target;

use super::training::build_dataset;

pub struct SplitArgs {
    pub dataset_path: String,
    pub train_out: String,
    pub val_out: String,
    pub ratio: f64,
}

fn write_csv(dataset: &Dataset, path: &str) -> Result<(), Box<dyn Error>> {
    let mut file = std::fs::File::create(path)?;

    let header: Vec<&str> = dataset.feature_names.iter().map(String::as_str).collect();
    writeln!(file, "{}", header.join(","))?;

    for row in 0..dataset.features.nrows() {
        let feature_cols: Vec<String> = (0..dataset.features.ncols())
            .map(|c| dataset.features[[row, c]].to_string())
            .collect();
        writeln!(file, "{},{}", dataset.labels[row], feature_cols.join(","))?;
    }

    Ok(())
}

pub fn run_split(args: &SplitArgs) -> Result<(), Box<dyn Error>> {
    if args.ratio <= 0.0 || args.ratio >= 1.0 {
        return Err(format!("--ratio must be in (0, 1), got {}", args.ratio).into());
    }

    let dataset = build_dataset(&args.dataset_path)?;
    let n = dataset.features.nrows();
    if n < 2 {
        return Err("dataset must have at least 2 rows to split".into());
    }

    let target = dataset.features.column(0).to_owned();
    let (train, val) = stratified_split_by_target(&dataset, &target, args.ratio, None);

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

