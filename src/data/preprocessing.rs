use ndarray::{Array1, Array2};

pub trait Normalizer {
    fn fit(&mut self, data: &Array2<f64>);
    fn transform(&self, data: &Array2<f64>) -> Array2<f64>;
    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64>;
}

#[derive(Debug, Clone, Default)]
pub struct StandardScaler {
    pub(crate) mean: Array1<f64>,
    pub(crate) std: Array1<f64>,
}

impl Normalizer for StandardScaler {
    fn fit(&mut self, data: &Array2<f64>) {
        self.mean = data.mean_axis(ndarray::Axis(0)).unwrap();
        self.std = data.std_axis(ndarray::Axis(0), 0.0).mapv(|v| v.max(1e-12));
    }

    fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        (data - &self.mean) / &self.std
    }

    fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }
}
