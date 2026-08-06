use mlp::data::loader::load_dataset;
use mlp::data::preprocessing::{Normalizer, StandardScaler};
use ndarray::arr2;

fn assert_matrix_close(
    actual: &ndarray::Array2<f64>,
    expected: &ndarray::Array2<f64>,
    tol: f64,
) {
    assert_eq!(actual.dim(), expected.dim());
    for ((i, j), value) in actual.indexed_iter() {
        let diff = (value - expected[[i, j]]).abs();
        assert!(
            diff <= tol,
            "mismatch at ({}, {}): actual={}, expected={}, diff={}",
            i,
            j,
            value,
            expected[[i, j]],
            diff
        );
    }
}

#[test]
fn standard_scaler_fit_transform_standardizes_columns() {
    let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let mut scaler = StandardScaler::default();

    let transformed = scaler.fit_transform(&data);

    let col_means = transformed
        .mean_axis(ndarray::Axis(0))
        .expect("mean along axis should exist");
    let col_stds = transformed.std_axis(ndarray::Axis(0), 0.0);

    for mean in col_means {
        assert!(mean.abs() < 1e-12);
    }
    for std in col_stds {
        assert!((std - 1.0).abs() < 1e-12);
    }
}

#[test]
fn standard_scaler_transform_uses_fitted_statistics() {
    let train = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let predict = arr2(&[[7.0, 8.0]]);
    let mut scaler = StandardScaler::default();

    scaler.fit(&train);
    let transformed = scaler.transform(&predict);

    let expected = arr2(&[[2.449489742783178, 2.449489742783178]]);
    assert_matrix_close(&transformed, &expected, 1e-12);
}

#[test]
fn standard_scaler_handles_real_dataset_from_data_csv() {
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

    let dataset =
        load_dataset(&csv_path, 1, names, 0).expect("loading data/data.csv should succeed");

    let mut scaler = StandardScaler::default();
    let scaled = scaler.fit_transform(&dataset.features);

    assert_eq!(scaled.dim(), dataset.features.dim());
    assert!(scaled.iter().all(|v| v.is_finite()));

    let col_means = scaled
        .mean_axis(ndarray::Axis(0))
        .expect("mean along axis should exist");
    assert!(col_means.iter().any(|m| m.abs() < 1e-10));
}
