use std::error::Error;

use mlp::data::loader::{Dataset, load_dataset};

fn build_dataset() -> Result<Dataset, Box<dyn Error>> {
    let base_features = vec![
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ];
    let stats = vec!["mean", "se", "extreme"];

    let mut names: Vec<String> = vec!["ID".to_string(), "Diagnosis".to_string()];
    for feature in &base_features {
        for stat in &stats {
            names.push(format!("{}_{}", feature, stat));
        }
    }

    let csv_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));
    load_dataset(&csv_path, 1, names, 0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let dataset = build_dataset()?;
    println!(
        "Loaded dataset: rows={}, cols={}",
        dataset.features.nrows(),
        dataset.features.ncols()
    );
    Ok(())
}
