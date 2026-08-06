use mlp::network::activation::{Activation, ActivationFunction};
use ndarray::{Array2, Axis, arr2};

fn assert_matrix_close(actual: &Array2<f64>, expected: &Array2<f64>, tol: f64) {
    assert_eq!(actual.dim(), expected.dim());
    for ((i, j), value) in actual.indexed_iter() {
        let diff = (value - expected[[i, j]]).abs();
        assert!(
            diff <= tol,
            "mismatch at ({}, {}): actual={}, expected={}, diff={}",
            i,
            j,
            value,
            expected[[i, j]],
            diff
        );
    }
}

#[test]
fn sigmoid_forward_and_derivative_are_correct() {
    let activation = ActivationFunction::Sigmoid;
    let x = arr2(&[[0.0, 1.0], [-1.0, 2.0]]);

    let y = activation.forward(&x);
    let ones = Array2::ones(y.dim());
    let dy = activation.backward(&y, &ones);

    let expected_y = arr2(&[
        [0.5, 0.7310585786300049],
        [0.2689414213699951, 0.8807970779778823],
    ]);
    let expected_dy = arr2(&[
        [0.25, 0.19661193324148185],
        [0.19661193324148185, 0.10499358540350662],
    ]);

    assert_matrix_close(&y, &expected_y, 1e-12);
    assert_matrix_close(&dy, &expected_dy, 1e-12);
}

#[test]
fn tanh_forward_and_derivative_are_correct() {
    let activation = ActivationFunction::Tanh;
    let x = arr2(&[[0.0, 1.0], [-1.0, 2.0]]);

    let y = activation.forward(&x);
    let ones = Array2::ones(y.dim());
    let dy = activation.backward(&y, &ones);

    let expected_y = arr2(&[
        [0.0, 0.7615941559557649],
        [-0.7615941559557649, 0.9640275800758169],
    ]);
    let expected_dy = arr2(&[
        [1.0, 0.41997434161402614],
        [0.41997434161402614, 0.07065082485316443],
    ]);

    assert_matrix_close(&y, &expected_y, 1e-12);
    assert_matrix_close(&dy, &expected_dy, 1e-12);
}

#[test]
fn relu_forward_derivative_and_backward_are_correct() {
    let activation = ActivationFunction::ReLU;
    let x = arr2(&[[-1.0, 0.0, 2.0]]);
    let grad = arr2(&[[2.0, 2.0, 2.0]]);

    let y = activation.forward(&x);
    let ones = Array2::ones(y.dim());
    let dy = activation.backward(&y, &ones);
    let back = activation.backward(&y, &grad);

    let expected_y = arr2(&[[0.0, 0.0, 2.0]]);
    let expected_dy = arr2(&[[0.0, 0.0, 1.0]]);
    let expected_back = arr2(&[[0.0, 0.0, 2.0]]);

    assert_matrix_close(&y, &expected_y, 1e-12);
    assert_matrix_close(&dy, &expected_dy, 1e-12);
    assert_matrix_close(&back, &expected_back, 1e-12);
}

#[test]
fn softmax_rows_sum_to_one() {
    let activation = ActivationFunction::Softmax;
    let x = arr2(&[[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]);

    let y = activation.forward(&x);
    let row_sums = y.sum_axis(Axis(1));

    for sum in row_sums.iter() {
        assert!((*sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn softmax_is_shift_invariant_per_row() {
    let activation = ActivationFunction::Softmax;
    let x = arr2(&[[1.0, 2.0, 3.0]]);
    let shifted = arr2(&[[101.0, 102.0, 103.0]]);

    let y = activation.forward(&x);
    let y_shifted = activation.forward(&shifted);

    assert_matrix_close(&y, &y_shifted, 1e-12);
}
