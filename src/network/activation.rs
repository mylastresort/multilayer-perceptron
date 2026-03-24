use ndarray::{Array2, Axis};

// List of activation functions that can be used in the neural network.
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax,
}

pub struct Sigmoid;
pub struct Tanh;
pub struct ReLU;
pub struct Softmax;

// Defines the trait for activation functions, which includes methods for forward and backward passes.
pub trait Activation {
    // Function to compute the activation function on the input data.
    fn as_function(&self, x: &Array2<f64>) -> Array2<f64>;
    // Derivative of the activation function for backpropagation.
    fn derivative(&self, x: &Array2<f64>) -> Array2<f64>;
    // Computes the forward pass of the activation function.
    fn forward(&self, x: &Array2<f64>) -> Array2<f64>;
    // Computes the backward pass (gradient) of the activation function.
    fn backward(&self, x: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64>;
}

impl Activation for Sigmoid {
    #[inline]
    fn as_function(&self, x: &Array2<f64>) -> Array2<f64> {
        1.0 / (1.0 + (-x).mapv(f64::exp))
    }

    #[inline]
    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        let sigmoid_x = self.as_function(x);
        &sigmoid_x * &(1.0 - &sigmoid_x)
    }

    #[inline]
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        self.as_function(x)
    }

    #[inline]
    fn backward(&self, x: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
        grad * self.derivative(x)
    }
}

impl Activation for Tanh {
    #[inline]
    fn as_function(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(f64::tanh)
    }

    #[inline]
    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        let tanh_x = self.as_function(x);
        tanh_x.mapv(|v| 1.0 - v * v)
    }

    #[inline]
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        self.as_function(x)
    }

    #[inline]
    fn backward(&self, x: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
        grad * self.derivative(x)
    }
}

impl Activation for ReLU {
    #[inline]
    fn as_function(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| v.max(0.0))
    }

    #[inline]
    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    #[inline]
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        self.as_function(x)
    }

    #[inline]
    fn backward(&self, x: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
        grad * self.derivative(x)
    }
}

impl Activation for Softmax {
    #[inline]
    fn as_function(&self, x: &Array2<f64>) -> Array2<f64> {
        let max = x.map_axis(Axis(1), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        let exp_x = (x - &max.insert_axis(Axis(1))).mapv(f64::exp);
        let sum_exp_x = exp_x.sum_axis(Axis(1));
        exp_x / &sum_exp_x.insert_axis(Axis(1))
    }

    #[inline]
    fn derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        let softmax_x = self.as_function(x);
        &softmax_x * &(1.0 - &softmax_x)
    }

    #[inline]
    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        self.as_function(x)
    }

    #[inline]
    fn backward(&self, x: &Array2<f64>, grad: &Array2<f64>) -> Array2<f64> {
        grad * self.derivative(x)
    }
}

#[cfg(test)]
mod tests {
    use super::{Activation, ReLU, Sigmoid, Softmax, Tanh};
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
        let activation = Sigmoid;
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
        let activation = Tanh;
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
        let activation = ReLU;
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
        let activation = Softmax;
        let x = arr2(&[[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]);

        let y = activation.forward(&x);
        let row_sums = y.sum_axis(Axis(1));

        for sum in row_sums.iter() {
            assert!((*sum - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn softmax_is_shift_invariant_per_row() {
        let activation = Softmax;
        let x = arr2(&[[1.0, 2.0, 3.0]]);
        let shifted = arr2(&[[101.0, 102.0, 103.0]]);

        let y = activation.forward(&x);
        let y_shifted = activation.forward(&shifted);

        assert_matrix_close(&y, &y_shifted, 1e-12);
    }
}
