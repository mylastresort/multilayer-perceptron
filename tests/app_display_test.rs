use mlp::app::display::{print_loaded_config, print_loaded_dataset};

#[test]
fn print_loaded_dataset_does_not_panic() {
    print_loaded_dataset("test_data.csv", 100, 31);
}

#[test]
fn print_loaded_config_does_not_panic() {
    print_loaded_config("config.yaml", 0.01, 84, 8, 4);
}
