use ndarray::Array2;
use ndarray_rand::{RandomExt, rand::distr::StandardUniform, rand_distr::Uniform};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightInitializer {
    Random,
    Xavier,
    He,
}

impl WeightInitializer {
    pub fn initialize(&self, rows: usize, cols: usize) -> Array2<f64> {
        match self {
            WeightInitializer::Random => Array2::random((rows, cols), StandardUniform),
            WeightInitializer::Xavier => {
                let limit = (6.0 / (rows + cols) as f64).sqrt();
                Array2::random(
                    (rows, cols),
                    Uniform::new(-limit, limit).expect("finite bounds cannot fail"),
                )
            }
            WeightInitializer::He => {
                let limit = (2.0 / rows as f64).sqrt();
                Array2::random(
                    (rows, cols),
                    Uniform::new(-limit, limit).expect("finite bounds cannot fail"),
                )
            }
        }
    }
}
