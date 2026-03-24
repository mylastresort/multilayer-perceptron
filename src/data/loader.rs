use std::collections::HashMap;
use std::io::{Error as IoError, ErrorKind};

use ndarray::{Array1, Array2};

// Dataset struct to hold the features, labels, and feature names
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    // 2D array for features (shape: [num_samples, num_features])
    pub features: Array2<f64>,
    // 1D array for labels (shape: [num_samples])
    pub labels: Array1<f64>,
    // Vector of feature names (length: num_features)
    pub feature_names: Vec<String>,
}

// Function to load the dataset from a CSV file - converts to hot-encoded categorical features by default
pub fn load_dataset(
    // Path to the CSV file containing the dataset
    _file_path: &str,
    // Number of rows to skip at the beginning of the file (e.g., for headers)
    _skiprows: usize,
    // Vector of feature names corresponding to the columns in the dataset (length should match num_features)
    _names: Vec<String>,
    // Index of the column to be used as labels (0-based index)
    _label_col: usize,
) -> Result<Dataset, Box<dyn std::error::Error>> {
    // use csv crate to read the CSV file and populate the Dataset struct
    // - Read the CSV file, skipping the specified number of rows
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false) // We will handle headers manually
        .flexible(true)
        .from_path(_file_path)?;

    // - Initialize vectors to hold the features and labels
    let mut features_vec: Vec<Vec<f64>> = Vec::new();
    let mut labels_vec: Vec<f64> = Vec::new();

    // hashmap to store the unique categorical values and their corresponding hot-encoded indices for each categorical column
    // use 2-level hashmap: column index -> (categorical value -> hot-encoded index)
    let mut categorical_map: HashMap<
        usize, // column index
        Vec<String>,
    > = HashMap::new();

    // - Iterate over the records in the CSV file)
    let mut expected_record_len: Option<usize> = None;
    for (i, result) in reader.records().enumerate() {
        // Skip the specified number of rows
        if i < _skiprows {
            continue;
        }
        let record = result?;
        if let Some(expected_len) = expected_record_len {
            if record.len() != expected_len {
                return Err(IoError::new(
                    ErrorKind::InvalidData,
                    format!(
                        "Inconsistent column count at data row {}: expected {}, got {}",
                        i - _skiprows + 1,
                        expected_len,
                        record.len()
                    ),
                )
                .into());
            }
        } else {
            expected_record_len = Some(record.len());
        }
        // - Extract the label from the specified column and convert it to f64
        let label: f64 = record[_label_col].parse()?;
        labels_vec.push(label);
        // - Extract the features from the remaining columns
        // if the column is categorical, convert it to hot-encoded features and then to ndarray format
        let mut features_row: Vec<f64> = Vec::new();
        for (j, value) in record.iter().enumerate() {
            if j == _label_col {
                continue; // Skip the label column
            }
            if value.trim().is_empty() {
                features_row.push(f64::default());
                continue;
            }
            // Check if the value is numeric or categorical
            if let Ok(num) = value.parse::<f64>() {
                features_row.push(num); // Numeric feature
            } else {
                // Categorical feature - convert to hot-encoded features
                // For simplicity, we will just use a hash of the value to create a unique feature
                let index = categorical_map
                    .entry(j)
                    .or_insert_with(Vec::new)
                    .iter()
                    .position(|v| v == &value)
                    .unwrap_or_else(|| {
                        // If the value is not already in the map, add it and return its index
                        categorical_map.get_mut(&j).unwrap().push(value.to_string());
                        categorical_map[&j].len() - 1
                    });
                features_row.push(index as f64); // Add the hot-encoded index as a feature
            }
        }
        features_vec.push(features_row);
    }

    // - Convert the features and labels vectors to ndarray format
    let num_samples = labels_vec.len();
    let num_features = features_vec.first().map_or(0, |row| row.len());
    let features_array = Array2::from_shape_vec(
        (num_samples, num_features),
        features_vec.into_iter().flatten().collect(),
    )?;
    let labels_array = Array1::from_vec(labels_vec);

    // - Create and return the Dataset struct
    Ok(Dataset {
        features: features_array,
        labels: labels_array,
        feature_names: _names,
    })
}

#[cfg(test)]
mod tests {
    use super::load_dataset;
    use ndarray::{Array1, Array2};
    use std::fs;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn load_dataset_from_data_csv() {
        let csv_path = format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR"));

        // Build the same 32-column schema used by the dataset: 2 id/label + 10 features x 3 stats.
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

        // Validate the generated schema length against the actual CSV data row width.
        let csv_contents = fs::read_to_string(&csv_path).expect("data/data.csv should be readable");
        let data_row = csv_contents
            .lines()
            .nth(1)
            .expect("data/data.csv should contain at least one data row");
        let data_col_count = data_row.split(',').count();
        assert_eq!(names.len(), data_col_count);

        let dataset =
            load_dataset(&csv_path, 1, names, 0).expect("loading data/data.csv should succeed");

        // Compile-time type checks for the returned ndarray outputs.
        fn assert_types(_features: &Array2<f64>, _labels: &Array1<f64>) {}
        assert_types(&dataset.features, &dataset.labels);

        // Sanity-check expected dimensions after skipping the header row.
        assert_eq!(dataset.labels.len(), 569);
        assert_eq!(dataset.features.nrows(), 569);
        assert_eq!(dataset.features.ncols(), 31);
        assert_eq!(dataset.feature_names.len(), 32);

        // Diagnosis is categorical and should produce exactly two encoded classes.
        let mut diagnosis_classes: Vec<i64> = dataset
            .features
            .column(0)
            .iter()
            .map(|v| *v as i64)
            .collect();
        diagnosis_classes.sort_unstable();
        diagnosis_classes.dedup();
        assert_eq!(diagnosis_classes.len(), 2);
        assert_eq!(diagnosis_classes, vec![0, 1]);

        // Spot-check a few known values from the first data row.
        assert_eq!(dataset.labels[0], 842302.0);
        assert_eq!(dataset.features[[0, 0]], 0.0);
        assert!((dataset.features[[0, 1]] - 17.99).abs() < 1e-12);
    }

    #[test]
    fn load_dataset_fails_on_extra_cells_in_row() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let csv_path = std::env::temp_dir().join(format!(
            "mlp_extra_cells_{}_{}.csv",
            process::id(),
            timestamp
        ));

        // Row 3 has one extra cell, so data rows do not have a consistent column count.
        let csv = "ID,Diagnosis,Radius\n1,M,10.0\n2,B,11.0,extra\n";
        fs::write(&csv_path, csv).expect("temporary csv should be writable");

        let result = load_dataset(
            csv_path
                .to_str()
                .expect("temporary csv path should be valid utf-8"),
            1,
            Vec::new(),
            0,
        );

        let _ = fs::remove_file(&csv_path);

        assert!(result.is_err());
        let err_msg = result.err().expect("error should exist").to_string();
        assert!(err_msg.contains("Inconsistent column count"));
    }

    #[test]
    fn load_dataset_defaults_missing_values_to_f64_default() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let csv_path = std::env::temp_dir().join(format!(
            "mlp_missing_values_{}_{}.csv",
            process::id(),
            timestamp
        ));

        // Row 3 keeps the same column count but has an empty numeric value for Radius.
        let csv = "ID,Diagnosis,Radius,Texture\n1,M,10.0,2.5\n2,B,,3.5\n";
        fs::write(&csv_path, csv).expect("temporary csv should be writable");

        let dataset = load_dataset(
            csv_path
                .to_str()
                .expect("temporary csv path should be valid utf-8"),
            1,
            Vec::new(),
            0,
        )
        .expect("loading csv with missing values should succeed");

        let _ = fs::remove_file(&csv_path);

        // Features are: Diagnosis (encoded), Radius, Texture.
        assert_eq!(dataset.features.nrows(), 2);
        assert_eq!(dataset.features.ncols(), 3);
        assert_eq!(dataset.features[[1, 1]], f64::default());
    }
}
