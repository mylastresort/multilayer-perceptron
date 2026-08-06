use mlp::network::config::NetworkConfig;
use mlp::training::loss::LossFunction;

#[test]
fn parses_yaml_loss_field() {
    let yaml = r#"
loss: binaryCrossentropy
input_layers:
  - size: 30
hidden_layers:
  - size: 24
  - size: 24
output_layers:
  - size: 2
"#;
    let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
    assert_eq!(config.loss, LossFunction::BinaryCrossEntropy);
}

#[test]
fn loss_defaults_to_binary_cross_entropy() {
    let yaml = r#"
input_layers:
  - size: 4
hidden_layers:
  - size: 4
  - size: 4
output_layers:
  - size: 2
"#;
    let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
    assert_eq!(config.loss, LossFunction::BinaryCrossEntropy);
}

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

#[test]
fn validates_zero_epochs() {
    let yaml = r#"
epochs: 0
input_layers:
  - size: 4
hidden_layers:
  - size: 4
  - size: 4
output_layers:
  - size: 2
"#;
    let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("epochs"),
        "unexpected error: {err}"
    );
}

#[test]
fn validates_zero_batch_size() {
    let yaml = r#"
batch_size: 0
input_layers:
  - size: 4
hidden_layers:
  - size: 4
  - size: 4
output_layers:
  - size: 2
"#;
    let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("batch_size"),
        "unexpected error: {err}"
    );
}

#[test]
fn validates_empty_input_layers() {
    let yaml = r#"
input_layers: []
hidden_layers:
  - size: 4
  - size: 4
output_layers:
  - size: 2
"#;
    let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("input layer"),
        "unexpected error: {err}"
    );
}

#[test]
fn validates_empty_output_layers() {
    let yaml = r#"
input_layers:
  - size: 4
hidden_layers:
  - size: 4
  - size: 4
output_layers: []
"#;
    let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("output layer"),
        "unexpected error: {err}"
    );
}

#[test]
fn validates_zero_size_layer() {
    let yaml = r#"
input_layers:
  - size: 0
hidden_layers:
  - size: 4
  - size: 4
output_layers:
  - size: 2
"#;
    let config: NetworkConfig = serde_yaml::from_str(yaml).expect("yaml should parse");
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("zero size"),
        "unexpected error: {err}"
    );
}

#[test]
fn from_yaml_file_loads_valid_config() {
    use std::io::Write;
    let path =
        std::env::temp_dir().join(format!("mlp_config_test_{}.yaml", std::process::id()));
    let yaml = r#"
learning_rate: 0.001
epochs: 5
batch_size: 16
input_layers:
  - size: 4
hidden_layers:
  - size: 8
  - size: 8
output_layers:
  - size: 2
"#;
    std::fs::File::create(&path)
        .unwrap()
        .write_all(yaml.as_bytes())
        .unwrap();
    let config = NetworkConfig::from_yaml_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(config.epochs, 5);
    assert_eq!(config.batch_size, 16);
}

#[test]
fn from_yaml_file_returns_error_for_missing_file() {
    let result = NetworkConfig::from_yaml_file("/tmp/nonexistent_mlp_config_xyz.yaml");
    assert!(result.is_err());
}
