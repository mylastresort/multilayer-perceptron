use ndarray::Array2;
use ndarray_rand::{RandomExt, rand::distributions::Standard, rand_distr::Uniform};
use serde::Deserialize;

// List of weight initialization methods for the network
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightInitializer {
    Random,
    Xavier,
    He,
}

// Implementation of the weight initialization methods
impl WeightInitializer {
    pub fn initialize(&self, rows: usize, cols: usize) -> Array2<f64> {
        match self {
            WeightInitializer::Random => Array2::random((rows, cols), Standard),
            WeightInitializer::Xavier => {
                let limit = (6.0 / (rows + cols) as f64).sqrt();
                Array2::random((rows, cols), Uniform::new(-limit, limit))
            }
            WeightInitializer::He => {
                let limit = (2.0 / rows as f64).sqrt();
                Array2::random((rows, cols), Uniform::new(-limit, limit))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::WeightInitializer;

    #[test]
    fn random_initializer_returns_expected_shape_and_range() {
        let rows = 32;
        let cols = 16;
        let weights = WeightInitializer::Random.initialize(rows, cols);

        assert_eq!(weights.dim(), (rows, cols));
        assert!(weights.iter().all(|&v| (0.0..1.0).contains(&v)));
    }

    #[test]
    fn xavier_initializer_values_are_within_calculated_limits() {
        let rows = 64;
        let cols = 32;
        let limit = (6.0 / (rows + cols) as f64).sqrt();
        let weights = WeightInitializer::Xavier.initialize(rows, cols);

        assert_eq!(weights.dim(), (rows, cols));
        assert!(weights.iter().all(|&v| v >= -limit && v < limit));
    }

    #[test]
    fn he_initializer_values_are_within_calculated_limits() {
        let rows = 128;
        let cols = 64;
        let limit = (2.0 / rows as f64).sqrt();
        let weights = WeightInitializer::He.initialize(rows, cols);

        assert_eq!(weights.dim(), (rows, cols));
        assert!(weights.iter().all(|&v| v >= -limit && v < limit));
    }
}
