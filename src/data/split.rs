use std::collections::BTreeMap;

use ndarray::{Array1, Axis};
use rand::{Rng, SeedableRng, prelude::SliceRandom, rngs::StdRng};

use crate::data::loader::Dataset;

fn split_and_collect_indices<R: Rng + ?Sized>(
    indices: &mut [usize],
    train_ratio: f64,
    rng: &mut R,
) -> (Vec<usize>, Vec<usize>) {
    indices.shuffle(rng);
    let train_size = (indices.len() as f64 * train_ratio).round() as usize;
    let (train, test) = indices.split_at(train_size);
    (train.to_vec(), test.to_vec())
}

fn split_groups_by_ratio<R: Rng + ?Sized>(
    groups: &BTreeMap<String, Vec<usize>>,
    train_ratio: f64,
    rng: &mut R,
) -> (Vec<usize>, Vec<usize>) {
    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();
    for group in groups.values() {
        let mut group_indices = group.clone();
        let (group_train, group_test) =
            split_and_collect_indices(&mut group_indices, train_ratio, rng);
        train_indices.extend(group_train);
        test_indices.extend(group_test);
    }
    (train_indices, test_indices)
}

fn select_dataset_rows(dataset: &Dataset, indices: &[usize]) -> Dataset {
    Dataset {
        features: dataset.features.select(Axis(0), indices),
        labels: dataset.labels.select(Axis(0), indices),
        feature_names: dataset.feature_names.clone(),
    }
}

fn rng_from_seed(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed_value) => StdRng::seed_from_u64(seed_value),
        None => StdRng::from_rng(&mut rand::rng()),
    }
}

/// Stratified split that preserves the class proportions of `target` in both
/// the training and validation sets.
pub fn stratified_split_by_target(
    dataset: &Dataset,
    target: &Array1<f64>,
    train_ratio: f64,
    seed: Option<u64>,
) -> (Dataset, Dataset) {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, value) in target.iter().enumerate() {
        groups.entry(value.to_string()).or_default().push(index);
    }

    let mut rng = rng_from_seed(seed);
    let (train_indices, test_indices) = split_groups_by_ratio(&groups, train_ratio, &mut rng);

    (
        select_dataset_rows(dataset, &train_indices),
        select_dataset_rows(dataset, &test_indices),
    )
}
