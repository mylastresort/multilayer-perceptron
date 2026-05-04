use ndarray::{Array1, Array2};

// This module contains data preprocessing utilities, such as normalization and standardization.
pub trait Normalizer {
    fn fit(&mut self, data: &Array2<f64>);
    fn transform(&self, data: &Array2<f64>) -> Array2<f64>;
    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64>;
}

// z-score normalization (standardization)
#[derive(Debug, Clone, Default)]
pub struct StandardScaler {
    mean: Array1<f64>,
    std: Array1<f64>,
}

// min-max normalization
#[derive(Debug, Clone, Default)]
pub struct MinMaxScaler {
    min: Array1<f64>,
    max: Array1<f64>,
}

// implementations of the Normalizer trait for both scalers
impl Normalizer for StandardScaler {
    fn fit(&mut self, data: &Array2<f64>) {
        self.mean = data.mean_axis(ndarray::Axis(0)).unwrap();
        self.std = data.std_axis(ndarray::Axis(0), 0.0);
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        // (x - u) / s
        (data - &self.mean) / &self.std
    }

    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}

// implementations of the Normalizer trait for MinMaxScaler
impl Normalizer for MinMaxScaler {
    fn fit(&mut self, data: &Array2<f64>) {
        self.min = data.fold_axis(ndarray::Axis(0), std::f64::INFINITY, |&a, &b| a.min(b));
        self.max = data.fold_axis(ndarray::Axis(0), std::f64::NEG_INFINITY, |&a, &b| a.max(b));
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        // (x - min) / (max - min)
        (data - &self.min) / (&self.max - &self.min)
    }

    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}

#[cfg(test)]
mod tests {
    use super::{MinMaxScaler, Normalizer, StandardScaler};
    use crate::data::loader::load_dataset;
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

        // For each column: mean = 0 and std = 1 after z-score normalization.
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
    fn min_max_scaler_scales_to_unit_interval() {
        let data = arr2(&[[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]);
        let mut scaler = MinMaxScaler::default();

        let transformed = scaler.fit_transform(&data);

        let expected = arr2(&[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]);
        assert_matrix_close(&transformed, &expected, 1e-12);
    }

    #[test]
    fn min_max_scaler_transform_uses_fitted_bounds() {
        let train = arr2(&[[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]);
        let predict = arr2(&[[3.0, 25.0]]);
        let mut scaler = MinMaxScaler::default();

        scaler.fit(&train);
        let transformed = scaler.transform(&predict);

        let expected = arr2(&[[0.5, 0.75]]);
        assert_matrix_close(&transformed, &expected, 1e-12);
    }

    #[test]
    fn min_max_scaler_fit_transform_matches_fit_then_transform() {
        let data = arr2(&[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]);

        let mut fit_then_transform_scaler = MinMaxScaler::default();
        fit_then_transform_scaler.fit(&data);
        let from_separate_calls = fit_then_transform_scaler.transform(&data);

        let mut fit_transform_scaler = MinMaxScaler::default();
        let from_fit_transform = fit_transform_scaler.fit_transform(&data);

        assert_matrix_close(&from_fit_transform, &from_separate_calls, 1e-12);
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

        // Scaling should preserve shape and produce finite numeric outputs.
        assert_eq!(scaled.dim(), dataset.features.dim());
        assert!(scaled.iter().all(|v| v.is_finite()));

        // At least one feature should be centered near zero.
        let col_means = scaled
            .mean_axis(ndarray::Axis(0))
            .expect("mean along axis should exist");
        assert!(col_means.iter().any(|m| m.abs() < 1e-10));
    }
}
