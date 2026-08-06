use crate::console::{Tone, bold, paint};

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
