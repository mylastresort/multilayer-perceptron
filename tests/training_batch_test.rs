use mlp::training::batch::create_batches;
use ndarray::{Array1, Array2};

fn data_4x2() -> (Array2<f64>, Array1<f64>) {
    let x = Array2::from_shape_fn((4, 2), |(i, j)| (i * 2 + j) as f64);
    let y = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
    (x, y)
}

#[test]
fn batch_iterator_visits_all_rows_once_with_batch_size_1() {
    let (x, y) = data_4x2();
    let indices: Vec<usize> = (0..4).collect();
    let mut iter = create_batches(x.view(), y.view(), &indices, 1);
    let mut count = 0;
    while iter.next_batch().is_some() {
        count += 1;
    }
    assert_eq!(count, 4);
}

#[test]
fn batch_iterator_groups_consecutive_indices_together() {
    let (x, y) = data_4x2();
    let indices: Vec<usize> = (0..4).collect();
    let mut iter = create_batches(x.view(), y.view(), &indices, 2);

    let (first, y_first) = iter.next_batch().unwrap();
    assert_eq!(first.dim(), (2, 2));
    assert_eq!(y_first.len(), 2);
    assert_eq!(first[[0, 0]], 0.0);
    assert_eq!(first[[1, 1]], 3.0);
    assert_eq!(y_first[0], 0.0);
    assert_eq!(y_first[1], 1.0);

    let (second, y_second) = iter.next_batch().unwrap();
    assert_eq!(second[[0, 1]], 5.0);
    assert_eq!(second[[1, 0]], 6.0);
    assert_eq!(y_second[0], 0.0);
    assert_eq!(y_second[1], 1.0);

    assert!(iter.next_batch().is_none());
}

#[test]
fn batch_iterator_sum_by_accumulates_all_rows() {
    let (x, y) = data_4x2();
    let indices: Vec<usize> = (0..4).collect();
    let mut iter = create_batches(x.view(), y.view(), &indices, 2);
    let total = iter.sum_by(|x_batch, _y| x_batch.nrows() as f64);
    assert_eq!(total, 4.0);
}

#[test]
fn batch_iterator_with_batch_size_larger_than_data() {
    let (x, y) = data_4x2();
    let indices: Vec<usize> = (0..4).collect();
    let mut iter = create_batches(x.view(), y.view(), &indices, 100);
    let mut count = 0;
    while iter.next_batch().is_some() {
        count += 1;
    }
    assert_eq!(count, 1);
}

#[test]
fn batch_iterator_empty_indices_returns_none_immediately() {
    let (x, y) = data_4x2();
    let indices: Vec<usize> = vec![];
    let mut iter = create_batches(x.view(), y.view(), &indices, 2);
    assert!(iter.next_batch().is_none());
}

#[test]
fn batch_iterator_each_batch_has_correct_shape() {
    let (x, y) = data_4x2();
    let indices: Vec<usize> = (0..4).collect();
    let mut iter = create_batches(x.view(), y.view(), &indices, 2);
    while let Some((x_batch, y_batch)) = iter.next_batch() {
        assert_eq!(x_batch.ncols(), 2);
        assert_eq!(x_batch.nrows(), y_batch.len());
    }
}

#[test]
fn batch_iterator_respects_shuffled_indices() {
    let (x, y) = data_4x2();

    let indices: Vec<usize> = (0..4).rev().collect();
    let mut iter = create_batches(x.view(), y.view(), &indices, 2);

    let (first, y_first) = iter.next_batch().unwrap();
    assert_eq!(first[[0, 0]], 6.0);
    assert_eq!(first[[1, 0]], 4.0);
    assert_eq!(y_first[0], 1.0);
    assert_eq!(y_first[1], 0.0);
}
