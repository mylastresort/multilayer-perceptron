use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use crate::network::activation::ActivationFunction;
use crate::network::initializer::WeightInitializer;
use crate::network::layer::Layer;
use crate::network::model::Network;
use crate::training::loss::LossFunction;
use crate::training::optimizer::OptimizerKind;

#[derive(Debug, Clone, Copy)]
pub enum LayerGroup {
    Input,
    Hidden,
    Output,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerTransitionSpec {
    pub from_size: usize,
    pub to_size: usize,
    pub to_group: LayerGroup,
    pub activation: ActivationFunction,
    pub initializer: WeightInitializer,
}

#[derive(Debug, Deserialize)]
pub struct NetworkConfig {
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default)]
    pub optimizer: OptimizerKind,
    #[serde(default)]
    pub loss: LossFunction,
    pub input_layers: Vec<LayerConfig>,
    pub hidden_layers: Vec<LayerConfig>,
    pub output_layers: Vec<LayerConfig>,
}

#[derive(Debug, Deserialize)]
pub struct LayerConfig {
    pub size: usize,
    pub activation: Option<ActivationFunction>,
    pub initializer: Option<WeightInitializer>,
}

fn default_learning_rate() -> f64 {
    0.0314
}

fn default_epochs() -> usize {
    84
}

fn default_batch_size() -> usize {
    8
}

fn default_hidden_activation() -> ActivationFunction {
    ActivationFunction::Sigmoid
}

fn default_output_activation() -> ActivationFunction {
    ActivationFunction::Softmax
}

fn default_initializer() -> WeightInitializer {
    WeightInitializer::He
}

fn default_activation_for(group: LayerGroup) -> ActivationFunction {
    match group {
        LayerGroup::Output => default_output_activation(),
        LayerGroup::Input | LayerGroup::Hidden => default_hidden_activation(),
    }
}

fn activation_for(group: LayerGroup, layer: &LayerConfig) -> ActivationFunction {
    layer
        .activation
        .unwrap_or_else(|| default_activation_for(group))
}

impl NetworkConfig {
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let raw = fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&raw)?;
        config.validate()?;
        Ok(config)
    }

    pub fn build_network(&self) -> Network {
        let mut builder = Network::builder()
            .learning_rate(self.learning_rate)
            .loss(self.loss);

        for spec in self.resolved_layer_specs() {
            builder = builder.add_layer(Layer::new(
                spec.from_size,
                spec.to_size,
                spec.activation,
                spec.initializer,
            ));
        }

        builder.build()
    }

    pub fn resolved_layer_specs(&self) -> Vec<LayerTransitionSpec> {
        let mut specs = Vec::new();
        for window in self.grouped_layers().windows(2) {
            let (_, current) = window[0];
            let (next_group, next) = window[1];
            specs.push(LayerTransitionSpec {
                from_size: current.size,
                to_size: next.size,
                to_group: next_group,
                activation: activation_for(next_group, next),
                initializer: next.initializer.unwrap_or_else(default_initializer),
            });
        }
        specs
    }

    fn grouped_layers(&self) -> Vec<(LayerGroup, &LayerConfig)> {
        self.input_layers
            .iter()
            .map(|layer| (LayerGroup::Input, layer))
            .chain(
                self.hidden_layers
                    .iter()
                    .map(|layer| (LayerGroup::Hidden, layer)),
            )
            .chain(
                self.output_layers
                    .iter()
                    .map(|layer| (LayerGroup::Output, layer)),
            )
            .collect()
    }

    pub fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.epochs == 0 {
            return Err("network config epochs must be greater than 0".into());
        }

        if self.batch_size == 0 {
            return Err("network config batch_size must be greater than 0".into());
        }

        if self.input_layers.is_empty() {
            return Err("network config must include at least one input layer size".into());
        }

        if self.hidden_layers.len() < 2 {
            return Err("network config must include at least 2 hidden layers".into());
        }

        if self.output_layers.is_empty() {
            return Err("network config must include at least one output layer size".into());
        }

        for (idx, layer) in self
            .input_layers
            .iter()
            .chain(self.hidden_layers.iter())
            .chain(self.output_layers.iter())
            .enumerate()
        {
            if layer.size == 0 {
                return Err(format!("layer {idx} has invalid zero size").into());
            }
        }

        Ok(())
    }
}
