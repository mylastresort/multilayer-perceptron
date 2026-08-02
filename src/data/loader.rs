use ndarray::{Array1, Array2};
use polars::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Dataset {
    pub features: Array2<f64>,
    pub labels: Array1<f64>,
    pub feature_names: Vec<String>,
}

type LoadResult<T> = Result<T, Box<dyn std::error::Error>>;

fn read_dataframe(file_path: &str, skiprows: usize) -> LoadResult<DataFrame> {
    let has_header = skiprows > 0;
    let df = CsvReadOptions::default()
        .with_has_header(has_header)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()?;

    if skiprows > 1 {
        let drop_count = skiprows - 1;
        let remaining = df.height().saturating_sub(drop_count);
        Ok(df.slice(drop_count as i64, remaining))
    } else {
        Ok(df)
    }
}

fn column_to_values(column: &Column) -> LoadResult<Vec<f64>> {
    if column.dtype() == &DataType::String {
        let str_ca = column.str()?;
        let values: Vec<Option<&str>> = str_ca.iter().collect();
        let mut classes: Vec<String> = values
            .iter()
            .filter_map(|opt_s| opt_s.filter(|s| !s.is_empty()).map(|s| s.to_string()))
            .collect();
        classes.sort_unstable();
        classes.dedup();
        Ok(values
            .iter()
            .map(|opt_s| match opt_s {
                Some(s) if !s.is_empty() => {
                    classes.iter().position(|c| c.as_str() == *s).unwrap() as f64
                }
                _ => f64::default(),
            })
            .collect())
    } else {
        Ok(column
            .cast(&DataType::Float64)?
            .f64()?
            .iter()
            .map(|opt| opt.unwrap_or_default())
            .collect())
    }
}

pub fn load_dataset(
    file_path: &str,
    skiprows: usize,
    names: Vec<String>,
    label_col: usize,
) -> LoadResult<Dataset> {
    let df = read_dataframe(file_path, skiprows)?;
    let n_rows = df.height();

    let columns: Vec<Vec<f64>> = df
        .get_column_names()
        .into_iter()
        .map(|col_name| column_to_values(df.column(col_name.as_str())?))
        .collect::<LoadResult<_>>()?;

    let labels_vec: Vec<f64> = columns[label_col].clone();
    let feature_cols: Vec<&Vec<f64>> = columns
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != label_col)
        .map(|(_, col)| col)
        .collect();

    let num_features = feature_cols.len();
    let features_flat: Vec<f64> = (0..n_rows)
        .flat_map(|row| feature_cols.iter().map(move |col| col[row]))
        .collect();

    let features_array = Array2::from_shape_vec((n_rows, num_features), features_flat)?;
    let labels_array = Array1::from_vec(labels_vec);

    Ok(Dataset {
        features: features_array,
        labels: labels_array,
        feature_names: names,
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

        let csv_contents = fs::read_to_string(&csv_path).expect("data/data.csv should be readable");
        let data_row = csv_contents
            .lines()
            .nth(1)
            .expect("data/data.csv should contain at least one data row");
        let data_col_count = data_row.split(',').count();
        assert_eq!(names.len(), data_col_count);

        let dataset =
            load_dataset(&csv_path, 0, names, 0).expect("loading data/data.csv should succeed");

        fn assert_types(_features: &Array2<f64>, _labels: &Array1<f64>) {}
        assert_types(&dataset.features, &dataset.labels);

        assert_eq!(dataset.labels.len(), 569);
        assert_eq!(dataset.features.nrows(), 569);
        assert_eq!(dataset.features.ncols(), 31);
        assert_eq!(dataset.feature_names.len(), 32);

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

        assert_eq!(dataset.labels[0], 842302.0);
        assert_eq!(dataset.features[[0, 0]], 1.0);
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

        assert_eq!(dataset.features.nrows(), 2);
        assert_eq!(dataset.features.ncols(), 3);
        assert_eq!(dataset.features[[1, 1]], f64::default());
    }

    #[test]
    fn load_dataset_skips_extra_rows_when_skiprows_greater_than_one() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let csv_path =
            std::env::temp_dir().join(format!("mlp_skiprows_{}_{}.csv", process::id(), timestamp));

        let csv = "A,B\nskip_this_row,999\n1.0,2.0\n3.0,4.0\n";
        fs::write(&csv_path, csv).expect("temporary csv should be writable");

        let result = load_dataset(
            csv_path
                .to_str()
                .expect("temporary csv path should be valid utf-8"),
            2,
            Vec::new(),
            0,
        );

        let _ = fs::remove_file(&csv_path);

        let dataset = result.expect("load with skiprows=2 should succeed");

        assert_eq!(dataset.features.nrows(), 2);
    }

    #[test]
    fn load_dataset_encodes_strings_by_sorted_order_not_first_appearance() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let csv_m_first = "ID,Diagnosis,Value\n1,M,10.0\n2,B,11.0\n";
        let csv_b_first = "ID,Diagnosis,Value\n2,B,11.0\n1,M,10.0\n";
        let path_m =
            std::env::temp_dir().join(format!("mlp_m_first_{}_{}.csv", process::id(), timestamp));
        let path_b =
            std::env::temp_dir().join(format!("mlp_b_first_{}_{}.csv", process::id(), timestamp));
        fs::write(&path_m, csv_m_first).expect("temp csv should be writable");
        fs::write(&path_b, csv_b_first).expect("temp csv should be writable");

        let m = load_dataset(path_m.to_str().unwrap(), 1, Vec::new(), 0).unwrap();
        let b = load_dataset(path_b.to_str().unwrap(), 1, Vec::new(), 0).unwrap();
        let _ = fs::remove_file(&path_m);
        let _ = fs::remove_file(&path_b);

        assert_eq!(m.features[[0, 0]], 1.0, "M row in M-first file");
        assert_eq!(m.features[[1, 0]], 0.0, "B row in M-first file");
        assert_eq!(b.features[[0, 0]], 0.0, "B row in B-first file");
        assert_eq!(b.features[[1, 0]], 1.0, "M row in B-first file");
    }

    #[test]
    fn load_dataset_handles_empty_string_in_string_column() {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        let csv_path = std::env::temp_dir().join(format!(
            "mlp_null_string_{}_{}.csv",
            process::id(),
            timestamp
        ));
        let csv = "ID,Diagnosis,Value\n1,M,10.0\n2,,11.0\n";
        fs::write(&csv_path, csv).expect("temporary csv should be writable");
        let result = load_dataset(
            csv_path.to_str().expect("path should be valid utf-8"),
            1,
            Vec::new(),
            0,
        );
        let _ = fs::remove_file(&csv_path);
        let dataset = result.expect("loading csv with empty string should succeed");
        assert_eq!(dataset.features.nrows(), 2);
    }
}
