use std::{error::Error, fs, path::Path};

use serde::Deserialize;

use crate::network::{
    activation::ActivationFunction, initializer::WeightInitializer, layer::Layer, model::Network,
};
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

impl NetworkConfig {
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let raw = fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&raw)?;
        config.validate()?;
        Ok(config)
    }

    pub fn build_network(&self) -> Network {
        let mut builder = Network::new().learning_rate(self.learning_rate);

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
        let mut groups: Vec<(LayerGroup, &LayerConfig)> = Vec::new();
        groups.extend(
            self.input_layers
                .iter()
                .map(|layer| (LayerGroup::Input, layer)),
        );
        groups.extend(
            self.hidden_layers
                .iter()
                .map(|layer| (LayerGroup::Hidden, layer)),
        );
        groups.extend(
            self.output_layers
                .iter()
                .map(|layer| (LayerGroup::Output, layer)),
        );

        let mut specs = Vec::new();
        for window in groups.windows(2) {
            let (_, current) = window[0];
            let (next_group, next) = window[1];

            let activation = next.activation.unwrap_or_else(|| match next_group {
                LayerGroup::Hidden => default_hidden_activation(),
                LayerGroup::Output => default_output_activation(),
                LayerGroup::Input => default_hidden_activation(),
            });

            let initializer = next.initializer.unwrap_or_else(default_initializer);

            specs.push(LayerTransitionSpec {
                from_size: current.size,
                to_size: next.size,
                to_group: next_group,
                activation,
                initializer,
            });
        }

        specs
    }

    fn validate(&self) -> Result<(), Box<dyn Error>> {
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
            return Err(
                "network config must include at least 2 hidden layers".into(),
            );
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

#[cfg(test)]
mod tests {
    use super::NetworkConfig;

    #[test]
    fn parses_yaml_network_config() {
        let yaml = r#"
learning_rate: 0.0314
epochs: 84
batch_size: 8
input_layers:
  - size: 30
hidden_layers:
  - size: 24
  - size: 24
  - size: 24
output_layers:
  - size: 2
"#;

        let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
        config.validate().expect("config should validate");

        let network = config.build_network();

        assert_eq!(network.layers.len(), 4);
        assert_eq!(config.epochs, 84);
        assert_eq!(config.batch_size, 8);
        assert!((config.learning_rate - 0.0314).abs() < 1e-12);
    }

    #[test]
    fn validates_at_least_two_hidden_layers() {
        let yaml_one_hidden = r#"
input_layers:
  - size: 30
hidden_layers:
  - size: 24
output_layers:
  - size: 2
"#;
        let config: NetworkConfig =
            serde_yaml::from_str(yaml_one_hidden).expect("yaml should parse");
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("at least 2 hidden layers"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validates_zero_hidden_layers() {
        let yaml_no_hidden = r#"
input_layers:
  - size: 30
hidden_layers: []
output_layers:
  - size: 2
"#;
        let config: NetworkConfig =
            serde_yaml::from_str(yaml_no_hidden).expect("yaml should parse");
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string().contains("at least 2 hidden layers"),
            "unexpected error: {err}"
        );
    }
}
