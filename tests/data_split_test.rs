use mlp::data::loader::Dataset;
use mlp::data::split::stratified_split_by_target;
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
fn stratified_split_by_target_preserves_class_proportions() {
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
    let target = dataset.features.column(0).to_owned();

    let (train, test) = stratified_split_by_target(&dataset, &target, 0.6, Some(13));

    let class_one_in = |ds: &Dataset| ds.features.column(0).iter().filter(|v| **v > 0.5).count();
    assert_eq!(train.features.nrows() + test.features.nrows(), 10);
    assert_eq!(class_one_in(&train), 3);
    assert_eq!(class_one_in(&test), 2);
    assert_eq!(train.features.nrows(), 6);
    assert_eq!(test.features.nrows(), 4);
}

#[test]
fn stratified_split_by_target_same_seed_same_split() {
    let dataset = build_dataset(
        array![[10.0], [11.0], [20.0], [21.0]],
        vec![0.0; 4],
        vec!["value"],
    );
    let target = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

    let (train_a, test_a) = stratified_split_by_target(&dataset, &target, 0.5, Some(7));
    let (train_b, test_b) = stratified_split_by_target(&dataset, &target, 0.5, Some(7));

    assert_eq!(train_a.features, train_b.features);
    assert_eq!(test_a.features, test_b.features);
}
