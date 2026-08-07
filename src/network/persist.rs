use std::{error::Error, fs, path::Path};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::data::preprocessing::StandardScaler;
use crate::network::activation::ActivationFunction;
use crate::network::layer::Layer;
use crate::network::model::Network;
use crate::training::loss::LossFunction;

#[derive(Serialize, Deserialize)]
struct SavedLayer {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
    activation: ActivationFunction,
}

#[derive(Serialize, Deserialize)]
struct SavedNetwork {
    learning_rate: f64,
    layers: Vec<SavedLayer>,
    #[serde(default)]
    feature_mean: Option<Vec<f64>>,
    #[serde(default)]
    feature_std: Option<Vec<f64>>,
    #[serde(default)]
    loss: LossFunction,
}

impl From<&Layer> for SavedLayer {
    fn from(layer: &Layer) -> Self {
        let weights: Vec<Vec<f64>> = layer.weights.outer_iter().map(|row| row.to_vec()).collect();
        Self {
            weights,
            bias: layer.bias.to_vec(),
            activation: layer.activation,
        }
    }
}

impl TryFrom<SavedLayer> for Layer {
    type Error = Box<dyn Error>;

    fn try_from(saved: SavedLayer) -> Result<Self, Self::Error> {
        let rows = saved.weights.len();
        if rows == 0 {
            return Err("saved layer has no weight rows".into());
        }
        let cols = saved.weights[0].len();
        if saved.bias.len() != cols {
            return Err(format!(
                "saved layer bias length {b} does not match weight columns {cols}",
                b = saved.bias.len()
            )
            .into());
        }

        let flat: Vec<f64> = saved.weights.into_iter().flatten().collect();
        let weights = Array2::from_shape_vec((rows, cols), flat)
            .map_err(|e| format!("failed to restore weight matrix: {e}"))?;
        let bias = Array1::from(saved.bias);

        use crate::network::initializer::WeightInitializer;
        let mut layer = Layer::new(rows, cols, saved.activation, WeightInitializer::He);
        layer.weights = weights;
        layer.bias = bias;
        Ok(layer)
    }
}

impl Network {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let saved = SavedNetwork {
            learning_rate: self.learning_rate,
            layers: self.layers.iter().map(SavedLayer::from).collect(),
            feature_mean: self.scaler.as_ref().map(|s| s.mean.to_vec()),
            feature_std: self.scaler.as_ref().map(|s| s.std.to_vec()),
            loss: self.loss,
        };
        let json = serde_json::to_string_pretty(&saved)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let json = fs::read_to_string(path)?;
        let saved: SavedNetwork = serde_json::from_str(&json)?;

        if saved.layers.len() < 3 {
            return Err(format!(
                "saved model has only {} layer(s); at least 3 are required (input→hidden×≥2→output)",
                saved.layers.len()
            )
            .into());
        }

        let mut layers: Vec<Layer> = Vec::with_capacity(saved.layers.len());
        for sl in saved.layers {
            layers.push(Layer::try_from(sl)?);
        }

        let scaler = match (saved.feature_mean, saved.feature_std) {
            (Some(mean), Some(std)) => Some(StandardScaler {
                mean: Array1::from_vec(mean),
                std: Array1::from_vec(std),
            }),
            _ => None,
        };

        Ok(Network {
            layers,
            learning_rate: saved.learning_rate,
            scaler,
            loss: saved.loss,
        })
    }
}
