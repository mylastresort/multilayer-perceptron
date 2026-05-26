mod app;

use std::error::Error;

use app::cli::{apply_net_overrides, parse_cli_args};
use app::display::{print_loaded_config, print_loaded_dataset, print_verbose_config};
use app::training::{build_dataset, train_from_dataset};
use mlp::network::config::NetworkConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let cli_args = parse_cli_args()?;
    let dataset = build_dataset(&cli_args.dataset_path)?;
    let mut network_config = NetworkConfig::from_yaml_file(&cli_args.config_path)?;
    apply_net_overrides(&mut network_config, &cli_args.net_overrides)?;
    let network = network_config.build_network();

    if cli_args.verbose {
        print_verbose_config(
            &network_config,
            &cli_args.dataset_path,
            &cli_args.config_path,
        );
    }

    print_loaded_dataset(
        &cli_args.dataset_path,
        dataset.features.nrows(),
        dataset.features.ncols(),
    );
    print_loaded_config(
        &cli_args.config_path,
        network.learning_rate,
        network_config.epochs,
        network_config.batch_size,
        network.layers.len(),
    );

    train_from_dataset(
        &dataset,
        &network_config,
        cli_args.gui,
        &cli_args.monitor_options,
    )?;

    Ok(())
}
