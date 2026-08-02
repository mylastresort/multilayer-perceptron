use std::collections::BTreeMap;

use ndarray::Axis;
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};

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

fn dataset_len(dataset: &Dataset) -> usize {
    dataset.features.nrows()
}

fn select_dataset_rows(dataset: &Dataset, indices: &[usize]) -> Dataset {
    Dataset {
        features: dataset.features.select(Axis(0), indices),
        labels: dataset.labels.select(Axis(0), indices),
        feature_names: dataset.feature_names.clone(),
    }
}

fn stratify_keys(dataset: &Dataset, stratify_col: &str) -> Vec<String> {
    if matches!(stratify_col, "label" | "labels") {
        return dataset
            .labels
            .iter()
            .map(|value| value.to_string())
            .collect();
    }

    let col_index = dataset
        .feature_names
        .iter()
        .position(|name| name == stratify_col)
        .expect("Stratification column not found in dataset feature names");

    let adjusted_index = if col_index < dataset.features.ncols() {
        col_index
    } else {
        col_index
            .checked_sub(1)
            .filter(|index| *index < dataset.features.ncols())
            .expect("Stratification column index is out of bounds")
    };

    (0..dataset_len(dataset))
        .map(|row| dataset.features[[row, adjusted_index]].to_string())
        .collect()
}

fn rng_from_seed(seed: Option<u64>) -> StdRng {
    match seed {
        Some(seed_value) => StdRng::seed_from_u64(seed_value),
        None => StdRng::from_rng(&mut rand::rng()),
    }
}

// Groups row indices by the stratification key, then splits each group.
fn stratified_split<R: Rng + ?Sized>(
    dataset: &Dataset,
    train_ratio: f64,
    stratify_col: &str,
    rng: &mut R,
) -> (Vec<usize>, Vec<usize>) {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let keys = stratify_keys(dataset, stratify_col);
    for (index, key) in keys.into_iter().enumerate() {
        groups.entry(key).or_default().push(index);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();
    for group in groups.values() {
        let mut group_indices = group.clone();
        let (group_train, group_test) = split_and_collect_indices(&mut group_indices, train_ratio, rng);
        train_indices.extend(group_train);
        test_indices.extend(group_test);
    }
    (train_indices, test_indices)
}

// Implements stratified train-test split for datasets.
pub fn train_test_split(
    dataset: &Dataset,
    train_ratio: f64,
    seed: Option<u64>,
    stratify: Option<&str>,
) -> (Dataset, Dataset) {
    let mut rng = rng_from_seed(seed);

    let (train_indices, test_indices) = match stratify {
        None => {
            let mut indices: Vec<usize> = (0..dataset_len(dataset)).collect();
            split_and_collect_indices(&mut indices, train_ratio, &mut rng)
        }
        Some(stratify_col) => stratified_split(dataset, train_ratio, stratify_col, &mut rng),
    };

    (
        select_dataset_rows(dataset, &train_indices),
        select_dataset_rows(dataset, &test_indices),
    )
}

#[cfg(test)]
mod tests {
    use super::train_test_split;
    use crate::data::loader::Dataset;
    use ndarray::{Array1, array};

    fn build_dataset(
        features: ndarray::Array2<f64>,
        labels: Vec<f64>,
        names: Vec<&str>,
    ) -> Dataset {
        Dataset {
            features,
            labels: Array1::from_vec(labels),
            feature_names: names.into_iter().map(str::to_string).collect(),
        }
    }

    #[test]
    fn random_split_preserves_total_rows() {
        let dataset = build_dataset(
            array![
                [0.0, 10.0],
                [1.0, 11.0],
                [0.0, 12.0],
                [1.0, 13.0],
                [0.0, 14.0],
                [1.0, 15.0],
                [0.0, 16.0],
                [1.0, 17.0],
                [0.0, 18.0],
                [1.0, 19.0]
            ],
            vec![0.0; 10],
            vec!["class", "value"],
        );

        let (train, test) = train_test_split(&dataset, 0.7, Some(7), None);

        assert_eq!(train.features.nrows(), 7);
        assert_eq!(test.features.nrows(), 3);
        assert_eq!(
            train.features.nrows() + test.features.nrows(),
            dataset.features.nrows()
        );
    }

    #[test]
    fn stratified_split_preserves_group_counts() {
        let dataset = build_dataset(
            array![
                [0.0, 10.0],
                [0.0, 11.0],
                [0.0, 12.0],
                [0.0, 13.0],
                [0.0, 14.0],
                [1.0, 20.0],
                [1.0, 21.0],
                [1.0, 22.0],
                [1.0, 23.0],
                [1.0, 24.0]
            ],
            vec![0.0; 10],
            vec!["class", "value"],
        );

        let (train, test) = train_test_split(&dataset, 0.6, Some(13), Some("class"));

        let train_class_zero = train
            .features
            .column(0)
            .iter()
            .filter(|value| (**value - 0.0).abs() < f64::EPSILON)
            .count();
        let train_class_one = train
            .features
            .column(0)
            .iter()
            .filter(|value| (**value - 1.0).abs() < f64::EPSILON)
            .count();
        let test_class_zero = test
            .features
            .column(0)
            .iter()
            .filter(|value| (**value - 0.0).abs() < f64::EPSILON)
            .count();
        let test_class_one = test
            .features
            .column(0)
            .iter()
            .filter(|value| (**value - 1.0).abs() < f64::EPSILON)
            .count();

        assert_eq!(train_class_zero, 3);
        assert_eq!(train_class_one, 3);
        assert_eq!(test_class_zero, 2);
        assert_eq!(test_class_one, 2);
    }

    #[test]
    fn stratified_split_handles_single_group_dataset() {
        let dataset = build_dataset(
            array![
                [1.0, 10.0],
                [1.0, 11.0],
                [1.0, 12.0],
                [1.0, 13.0],
                [1.0, 14.0]
            ],
            vec![1.0; 5],
            vec!["class", "value"],
        );

        let (train, test) = train_test_split(&dataset, 0.8, Some(99), Some("class"));

        assert_eq!(train.features.nrows(), 4);
        assert_eq!(test.features.nrows(), 1);
        assert!(
            train
                .features
                .column(0)
                .iter()
                .all(|value| (*value - 1.0).abs() < f64::EPSILON)
        );
        assert!(
            test.features
                .column(0)
                .iter()
                .all(|value| (*value - 1.0).abs() < f64::EPSILON)
        );
    }

    #[test]
    fn same_seed_produces_same_split() {
        let dataset = build_dataset(
            array![
                [0.0, 10.0],
                [0.0, 11.0],
                [0.0, 12.0],
                [0.0, 13.0],
                [1.0, 20.0],
                [1.0, 21.0],
                [1.0, 22.0],
                [1.0, 23.0]
            ],
            vec![0.0; 8],
            vec!["class", "value"],
        );

        let (train_a, test_a) = train_test_split(&dataset, 0.75, Some(1234), Some("class"));
        let (train_b, test_b) = train_test_split(&dataset, 0.75, Some(1234), Some("class"));

        assert_eq!(train_a.features, train_b.features);
        assert_eq!(train_a.labels, train_b.labels);
        assert_eq!(test_a.features, test_b.features);
        assert_eq!(test_a.labels, test_b.labels);
    }

    #[test]
    fn stratified_by_label_column_uses_labels_array() {
        let dataset = build_dataset(
            array![
                [10.0],
                [11.0],
                [12.0],
                [13.0],
                [20.0],
                [21.0],
                [22.0],
                [23.0]
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec!["value"],
        );

        let (train, test) = train_test_split(&dataset, 0.75, Some(42), Some("label"));
        assert_eq!(train.features.nrows() + test.features.nrows(), 8);

        // Both classes should appear in the training set
        let train_labels: std::collections::HashSet<i64> =
            train.labels.iter().map(|v| *v as i64).collect();
        assert!(train_labels.contains(&0));
        assert!(train_labels.contains(&1));
    }

    #[test]
    fn stratified_by_labels_alias_works() {
        let dataset = build_dataset(
            array![[1.0], [2.0], [3.0], [4.0]],
            vec![0.0, 0.0, 1.0, 1.0],
            vec!["feat"],
        );
        let (train, test) = train_test_split(&dataset, 0.5, Some(7), Some("labels"));
        assert_eq!(train.features.nrows() + test.features.nrows(), 4);
    }

    #[test]
    fn random_split_no_seed_completes_without_panic() {
        let dataset = build_dataset(
            array![[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]],
            vec![0.0, 1.0, 0.0, 1.0],
            vec!["a", "b"],
        );
        let (train, test) = train_test_split(&dataset, 0.75, None, None);
        assert_eq!(train.features.nrows() + test.features.nrows(), 4);
    }

    #[test]
    fn stratified_split_uses_adjusted_column_index_when_names_exceed_ncols() {
        // feature_names has 3 entries but features has only 2 columns.
        // "extra" is at position 2 in feature_names; col_index(2) >= ncols(2)
        // triggers the else-branch: adjusted_index = col_index.checked_sub(1) = 1.
        let dataset = build_dataset(
            array![[0.0, 10.0], [0.0, 11.0], [1.0, 20.0], [1.0, 21.0]],
            vec![0.0, 0.0, 1.0, 1.0],
            vec!["class", "value", "extra"],
        );
        let (train, test) = train_test_split(&dataset, 0.5, Some(1), Some("extra"));
        assert_eq!(train.features.nrows() + test.features.nrows(), 4);
    }
}
