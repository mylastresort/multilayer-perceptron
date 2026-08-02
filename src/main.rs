mod app;

use std::error::Error;

use app::cli::{apply_net_overrides, parse_env_args};
use app::display::{print_loaded_config, print_loaded_dataset, print_verbose_config};
use app::predict::{PredictArgs, run_predict};
use app::split::{SplitArgs, run_split};
use app::training::{build_dataset, train_from_dataset};
use app::types::{CliArgs, Subcommand};
use mlp::data::loader::Dataset;
use mlp::network::config::NetworkConfig;
use mlp::network::model::Network;

fn main() -> Result<(), Box<dyn Error>> {
    let cli_args = parse_env_args()?;

    match &cli_args.subcommand {
        Subcommand::Split => handle_split(&cli_args)?,
        Subcommand::Train => handle_train(&cli_args)?,
        Subcommand::Predict => handle_predict(&cli_args)?,
    }
    Ok(())
}

fn handle_split(cli_args: &CliArgs) -> Result<(), Box<dyn Error>> {
    let train_out = cli_args
        .train_out
        .clone()
        .ok_or("--train-out <PATH> is required for the split subcommand")?;
    let val_out = cli_args
        .val_out
        .clone()
        .ok_or("--val-out <PATH> is required for the split subcommand")?;
    run_split(&SplitArgs {
        dataset_path: cli_args.dataset_path.clone(),
        train_out,
        val_out,
        ratio: cli_args.split_ratio,
    })
}

fn print_train_setup(
    cli_args: &CliArgs,
    dataset: &Dataset,
    network: &Network,
    network_config: &NetworkConfig,
) {
    if cli_args.verbose {
        print_verbose_config(
            network_config,
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
}

fn handle_train(cli_args: &CliArgs) -> Result<(), Box<dyn Error>> {
    let dataset = build_dataset(&cli_args.dataset_path)?;
    let mut network_config = NetworkConfig::from_yaml_file(&cli_args.config_path)?;
    apply_net_overrides(&mut network_config, &cli_args.net_overrides)?;
    let network = network_config.build_network();

    print_train_setup(cli_args, &dataset, &network, &network_config);

    let config_stem = std::path::Path::new(&cli_args.config_path)
        .file_stem()
        .and_then(|stem| stem.to_str());

    train_from_dataset(
        &dataset,
        &network_config,
        &cli_args.monitor_options,
        config_stem,
        cli_args.model_out.as_deref().map(std::path::Path::new),
    )?;
    Ok(())
}

fn handle_predict(cli_args: &CliArgs) -> Result<(), Box<dyn Error>> {
    let model_path = cli_args
        .model_in
        .clone()
        .ok_or("--model <PATH> is required for the predict subcommand")?;
    run_predict(&PredictArgs {
        dataset_path: cli_args.dataset_path.clone(),
        model_path,
    })
}
