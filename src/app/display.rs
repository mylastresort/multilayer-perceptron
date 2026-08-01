use mlp::console::{Tone, bold, paint};
use mlp::network::config::{LayerGroup, NetworkConfig};
use mlp::network::{activation::ActivationFunction, initializer::WeightInitializer};

fn activation_label(activation: ActivationFunction) -> &'static str {
    match activation {
        ActivationFunction::Sigmoid => "sigmoid",
        ActivationFunction::Tanh => "tanh",
        ActivationFunction::ReLU => "relu",
        ActivationFunction::Softmax => "softmax",
    }
}

fn initializer_label(initializer: WeightInitializer) -> &'static str {
    match initializer {
        WeightInitializer::Random => "random",
        WeightInitializer::Xavier => "xavier",
        WeightInitializer::He => "he",
    }
}

fn group_label(group: LayerGroup) -> &'static str {
    match group {
        LayerGroup::Input => "input",
        LayerGroup::Hidden => "hidden",
        LayerGroup::Output => "output",
    }
}

pub fn print_verbose_config(config: &NetworkConfig, dataset_path: &str, config_path: &str) {
    println!(
        "{}",
        paint(
            "+------------------------------------------------------------+",
            Tone::Accent
        )
    );
    println!(
        "{}",
        bold(&paint(
            "|                 MLP LOADED CONFIG SUMMARY                  |",
            Tone::Accent
        ))
    );
    println!(
        "{}",
        paint(
            "+------------------------------------------------------------+",
            Tone::Accent
        )
    );
    println!(" {} {}", paint("Dataset path :", Tone::Info), dataset_path);
    println!(" {} {}", paint("Config path  :", Tone::Info), config_path);
    println!(
        " {} {:.6}",
        paint("Learning rate:", Tone::Info),
        config.learning_rate
    );
    println!(" {} {}", paint("Epochs       :", Tone::Info), config.epochs);
    println!(
        " {} {}",
        paint("Batch size   :", Tone::Info),
        config.batch_size
    );

    let input_sizes: Vec<usize> = config.input_layers.iter().map(|layer| layer.size).collect();
    let hidden_sizes: Vec<usize> = config
        .hidden_layers
        .iter()
        .map(|layer| layer.size)
        .collect();
    let output_sizes: Vec<usize> = config
        .output_layers
        .iter()
        .map(|layer| layer.size)
        .collect();
    println!(
        " {} {:?}",
        paint("Input layers :", Tone::TrainMetric),
        input_sizes
    );
    println!(
        " {} {:?}",
        paint("Hidden layers:", Tone::TrainMetric),
        hidden_sizes
    );
    println!(
        " {} {:?}",
        paint("Output layers:", Tone::ValMetric),
        output_sizes
    );

    println!(
        "{}",
        paint(
            "--------------------------------------------------------------",
            Tone::Muted
        )
    );
    println!(
        " {}",
        bold(&paint("Resolved transitions (with defaults):", Tone::Info))
    );

    for (idx, spec) in config.resolved_layer_specs().iter().enumerate() {
        println!(
            "  {:>2}. {:>3} -> {:>3} | {}={} | {}={} | {}={}",
            idx + 1,
            spec.from_size,
            spec.to_size,
            paint("to", Tone::Muted),
            group_label(spec.to_group),
            paint("activation", Tone::Muted),
            activation_label(spec.activation),
            paint("initializer", Tone::Muted),
            initializer_label(spec.initializer)
        );
    }

    println!(
        "{}",
        paint(
            "+------------------------------------------------------------+",
            Tone::Accent
        )
    );
}

pub fn print_loaded_dataset(dataset_path: &str, rows: usize, cols: usize) {
    println!(
        "{} {} {}",
        bold(&paint("Loaded dataset:", Tone::Info)),
        dataset_path,
        paint(&format!("(rows={}, cols={})", rows, cols), Tone::Muted)
    );
}

pub fn print_loaded_config(
    config_path: &str,
    learning_rate: f64,
    epochs: usize,
    batch_size: usize,
    layers: usize,
) {
    println!(
        "{} {} {}",
        bold(&paint("Loaded config:", Tone::Info)),
        config_path,
        paint(
            &format!(
                "(learning_rate={}, epochs={}, batch_size={}, layers={})",
                learning_rate, epochs, batch_size, layers
            ),
            Tone::Muted
        )
    );
}

#[cfg(test)]
mod tests {
    use super::{
        activation_label, group_label, initializer_label, print_loaded_config,
        print_loaded_dataset, print_verbose_config,
    };
    use mlp::network::{
        activation::ActivationFunction, config::LayerGroup, initializer::WeightInitializer,
    };

    fn minimal_config() -> mlp::network::config::NetworkConfig {
        let yaml = r#"
learning_rate: 0.01
epochs: 1
batch_size: 8
input_layers:
  - size: 4
hidden_layers:
  - size: 4
  - size: 4
output_layers:
  - size: 2
"#;
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn activation_label_covers_all_variants() {
        assert_eq!(activation_label(ActivationFunction::Sigmoid), "sigmoid");
        assert_eq!(activation_label(ActivationFunction::Tanh), "tanh");
        assert_eq!(activation_label(ActivationFunction::ReLU), "relu");
        assert_eq!(activation_label(ActivationFunction::Softmax), "softmax");
    }

    #[test]
    fn initializer_label_covers_all_variants() {
        assert_eq!(initializer_label(WeightInitializer::Random), "random");
        assert_eq!(initializer_label(WeightInitializer::Xavier), "xavier");
        assert_eq!(initializer_label(WeightInitializer::He), "he");
    }

    #[test]
    fn group_label_covers_all_variants() {
        assert_eq!(group_label(LayerGroup::Input), "input");
        assert_eq!(group_label(LayerGroup::Hidden), "hidden");
        assert_eq!(group_label(LayerGroup::Output), "output");
    }

    #[test]
    fn print_loaded_dataset_does_not_panic() {
        print_loaded_dataset("test_data.csv", 100, 31);
    }

    #[test]
    fn print_loaded_config_does_not_panic() {
        print_loaded_config("config.yaml", 0.01, 84, 8, 4);
    }

    #[test]
    fn print_verbose_config_does_not_panic() {
        let config = minimal_config();
        print_verbose_config(&config, "test_data.csv", "config.yaml");
    }
}
