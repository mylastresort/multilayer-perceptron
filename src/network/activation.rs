use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

// List of activation functions that can be used in the neural network.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    #[serde(alias = "relu", alias = "ReLU")]
    ReLU,
    Softmax,
}

// Defines the trait for activation functions, which includes methods for forward and backward passes.
pub trait Activation {
    // Function to compute the activation function on the input data.
    fn as_function(&self, x: &Array2<f64>) -> Array2<f64>;
    // Derivative of the activation function for backpropagation.
    fn derivative(&self, x: &Array2<f64>) -> Array2<f64>;
    // Computes the forward pass of the activation function.
    #[inline]
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        self.as_function(x)
    }
    // Computes the backward pass (gradient) of the activation function.
    #[inline]
    fn backward(&self, x: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
        grad * self.derivative(x)
    }
}

impl Activation for ActivationFunction {
    #[inline]
    fn as_function(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            ActivationFunction::Tanh => x.mapv(f64::tanh),
            ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
            ActivationFunction::Softmax => {
                let max = x.map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
                let exp_x = (x - &max.insert_axis(Axis(1))).mapv(f64::exp);
                let sum_exp_x = exp_x.sum_axis(Axis(1));
                exp_x / &sum_exp_x.insert_axis(Axis(1))
            }
        }
    }

    #[inline]
    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFunction::Sigmoid => {
                let sigmoid_x = self.as_function(x);
                sigmoid_x.mapv(|v| v * (1.0 - v))
            }
            ActivationFunction::Tanh => {
                let tanh_x = self.as_function(x);
                tanh_x.mapv(|v| 1.0 - v * v)
            }
            ActivationFunction::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            ActivationFunction::Softmax => {
                let softmax_x = self.as_function(x);
                &softmax_x * &(1.0 - &softmax_x)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Activation, ActivationFunction};
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
        let dy = activation.derivative(&x);

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
        let dy = activation.derivative(&x);

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
        let dy = activation.derivative(&x);
        let back = activation.backward(&x, &grad);

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

    #[test]
    fn softmax_derivative_is_elementwise_s_times_one_minus_s() {
        let activation = ActivationFunction::Softmax;
        let x = arr2(&[[1.0, 2.0, 3.0]]);

        let s = activation.forward(&x);
        let dy = activation.derivative(&x);

        // derivative = s * (1 - s) elementwise
        let expected = &s * &(1.0 - &s);
        assert_matrix_close(&dy, &expected, 1e-12);
    }
}
