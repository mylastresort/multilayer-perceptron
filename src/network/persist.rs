//! Model serialisation / deserialisation using JSON.
//!
//! Saved format (one JSON object):
//! ```json
//! {
//!   "learning_rate": 0.01,
//!   "layers": [
//!     { "weights": [[…], …], "bias": […], "activation": "sigmoid" },
//!     …
//!   ]
//! }
//! ```
//! The weight initialiser is **not** stored because weights are already
//! initialised; only the learned values matter.

use std::{error::Error, fs, path::Path};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::network::{activation::ActivationFunction, layer::Layer, model::Network};

// ---------------------------------------------------------------------------
// On-disk representation
// ---------------------------------------------------------------------------

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
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

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

        // Use a no-op initialiser placeholder; weights are restored directly.
        use crate::network::initializer::WeightInitializer;
        let mut layer = Layer::new(rows, cols, saved.activation, WeightInitializer::He);
        layer.weights = weights;
        layer.bias = bias;
        Ok(layer)
    }
}

// ---------------------------------------------------------------------------
// Public API on Network
// ---------------------------------------------------------------------------

impl Network {
    /// Serialise the network (topology + learned weights) to a JSON file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let saved = SavedNetwork {
            learning_rate: self.learning_rate,
            layers: self.layers.iter().map(SavedLayer::from).collect(),
            feature_mean: self.feature_mean.as_ref().map(|m| m.to_vec()),
            feature_std: self.feature_std.as_ref().map(|s| s.to_vec()),
        };
        let json = serde_json::to_string_pretty(&saved)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Deserialise a network previously saved with [`Network::save`].
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

        let feature_mean = saved.feature_mean.map(|v| {
            use ndarray::Array1;
            Array1::from_vec(v)
        });
        let feature_std = saved.feature_std.map(|v| {
            use ndarray::Array1;
            Array1::from_vec(v)
        });

        Ok(Network {
            layers,
            learning_rate: saved.learning_rate,
            feature_mean,
            feature_std,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{
        activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
    };

    fn three_layer_net() -> Network {
        Network::new()
            .add_layer(Layer::new(
                2,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                4,
                ActivationFunction::Tanh,
                WeightInitializer::Xavier,
            ))
            .add_layer(Layer::new(
                4,
                1,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .build()
    }

    #[test]
    fn load_nonexistent_file_returns_error() {
        let result = Network::load("/tmp/this_file_does_not_exist_mlp.json");
        assert!(result.is_err());
    }

    #[test]
    fn load_invalid_json_returns_error() {
        let path =
            std::env::temp_dir().join(format!("mlp_invalid_json_{}.json", std::process::id()));
        std::fs::write(&path, "not valid json {{").unwrap();
        let result = Network::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn load_model_with_fewer_than_three_layers_returns_error() {
        // Build a JSON with only 2 layers (bypassing Network builder)
        let json = serde_json::json!({
            "learning_rate": 0.01,
            "layers": [
                {
                    "weights": [[0.1, 0.2]],
                    "bias": [0.0, 0.0],
                    "activation": "sigmoid"
                },
                {
                    "weights": [[0.3], [0.4]],
                    "bias": [0.0],
                    "activation": "sigmoid"
                }
            ]
        })
        .to_string();

        let path = std::env::temp_dir().join(format!("mlp_two_layers_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();
        let result = Network::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn load_layer_with_empty_weights_returns_error() {
        let json = serde_json::json!({
            "learning_rate": 0.01,
            "layers": [
                { "weights": [], "bias": [], "activation": "sigmoid" },
                { "weights": [[0.1]], "bias": [0.0], "activation": "sigmoid" },
                { "weights": [[0.2]], "bias": [0.0], "activation": "sigmoid" }
            ]
        })
        .to_string();

        let path =
            std::env::temp_dir().join(format!("mlp_empty_weights_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();
        let result = Network::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn load_layer_with_mismatched_bias_length_returns_error() {
        // First layer: weights are 2×2 (2 columns), but bias has 3 elements.
        let json = serde_json::json!({
            "learning_rate": 0.01,
            "layers": [
                {
                    "weights": [[0.1, 0.2], [0.3, 0.4]],
                    "bias": [0.0, 0.0, 0.0],
                    "activation": "sigmoid"
                },
                {
                    "weights": [[0.3], [0.4]],
                    "bias": [0.0],
                    "activation": "sigmoid"
                },
                {
                    "weights": [[0.5]],
                    "bias": [0.0],
                    "activation": "sigmoid"
                }
            ]
        })
        .to_string();

        let path =
            std::env::temp_dir().join(format!("mlp_bias_mismatch_{}.json", std::process::id()));
        std::fs::write(&path, json).unwrap();
        let result = Network::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(result.is_err());
        let Err(e) = result else { unreachable!() };
        let msg = e.to_string();
        assert!(msg.contains("bias"), "unexpected error: {msg}");
    }

    #[test]
    fn save_and_load_roundtrip_three_layer_net() {
        let net = three_layer_net();
        let path =
            std::env::temp_dir().join(format!("mlp_persist_unit_{}.json", std::process::id()));
        net.save(&path).unwrap();
        let loaded = Network::load(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.layers.len(), 3);
        assert!((loaded.learning_rate - net.learning_rate).abs() < 1e-12);
    }
}
