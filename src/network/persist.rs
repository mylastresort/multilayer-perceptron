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
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl From<&Layer> for SavedLayer {
    fn from(layer: &Layer) -> Self {
        let weights: Vec<Vec<f64>> = layer
            .weights
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
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

        Ok(Network {
            layers,
            learning_rate: saved.learning_rate,
        })
    }
}
