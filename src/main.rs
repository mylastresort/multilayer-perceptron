mod app;

use std::error::Error;

use app::cli::{apply_net_overrides, parse_args, usage};
use app::display::{print_loaded_config, print_loaded_dataset, print_verbose_config};
use app::predict::{PredictArgs, run_predict};
use app::split::{SplitArgs, run_split};
use app::training::{build_dataset, train_from_dataset};
use app::types::Subcommand;
use mlp::network::config::NetworkConfig;

fn main() -> Result<(), Box<dyn Error>> {
    let mut env_args = std::env::args();
    let binary_name = env_args.next().unwrap_or_else(|| "mlp".to_string());
    let rest: Vec<String> = env_args.collect();

    let is_help = rest.is_empty()
        || rest[0] == "--help"
        || rest[0] == "-h"
        || rest.iter().any(|a| a == "--help" || a == "-h");

    let cli_args = match parse_args(&binary_name, &rest) {
        Err(e) if is_help => {
            println!("{e}");
            std::process::exit(0);
        }
        other => other?,
    };

    match cli_args.subcommand {
        // ---------------------------------------------------------------
        // split: export train/val CSV files
        // ---------------------------------------------------------------
        Subcommand::Split => {
            let train_out = cli_args
                .train_out
                .ok_or("--train-out <PATH> is required for the split subcommand")?;
            let val_out = cli_args
                .val_out
                .ok_or("--val-out <PATH> is required for the split subcommand")?;
            run_split(&SplitArgs {
                dataset_path: cli_args.dataset_path,
                train_out,
                val_out,
                ratio: cli_args.split_ratio,
            })?;
        }

        // ---------------------------------------------------------------
        // train: fit network and save model
        // ---------------------------------------------------------------
        Subcommand::Train => {
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

            let trained = train_from_dataset(
                &dataset,
                &network_config,
                cli_args.gui,
                &cli_args.monitor_options,
            )?;

            if let Some(model_path) = &cli_args.model_out {
                trained.save(model_path)?;
                println!("Model saved to {model_path}");
            }
        }

        // ---------------------------------------------------------------
        // predict: load model, run inference, report binary cross-entropy
        // ---------------------------------------------------------------
        Subcommand::Predict => {
            let model_path = cli_args
                .model_in
                .ok_or("--model <PATH> is required for the predict subcommand")?;
            run_predict(&PredictArgs {
                dataset_path: cli_args.dataset_path,
                model_path,
            })?;
        }
    }

    Ok(())
}
