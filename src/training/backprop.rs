use ndarray::{Array1, Array2};

use crate::network::model::Network;

#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl From<(Array2<f64>, Array1<f64>)> for LayerGradients {
    fn from((weights, bias): (Array2<f64>, Array1<f64>)) -> Self {
        Self { weights, bias }
    }
}

impl Network {
    pub fn backward(&mut self, _upstream_grad: Array2<f64>) -> Vec<LayerGradients> {
        let mut upstream_grad = _upstream_grad;
        let mut gd = Vec::with_capacity(self.layers.len());
        for l in self.layers.iter().rev() {
            let (g_input, g_weights, g_bias) = l.backward(&upstream_grad);
            gd.push(LayerGradients::from((g_weights, g_bias)));
            upstream_grad = g_input;
        }

        gd.reverse();
        gd
    }
}

#[cfg(test)]
mod tests {
    use super::LayerGradients;
    use crate::network::{
        activation::ActivationFunction, initializer::WeightInitializer, layer::Layer,
        model::Network,
    };
    use crate::training::loss::{Loss, LossFunction};
    use ndarray::{Array1, Array2, arr1, arr2};

    fn tiny_net() -> Network {
        Network::new()
            .add_layer(Layer::new(
                2,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .add_layer(Layer::new(
                4,
                1,
                ActivationFunction::Sigmoid,
                WeightInitializer::He,
            ))
            .build()
    }

    #[test]
    fn layer_gradients_from_tuple() {
        let w = Array2::zeros((3, 4));
        let b = Array1::zeros(4);
        let lg = LayerGradients::from((w.clone(), b.clone()));
        assert_eq!(lg.weights.dim(), w.dim());
        assert_eq!(lg.bias.len(), b.len());
    }

    #[test]
    fn backward_returns_one_gradient_per_layer() {
        let mut net = tiny_net();
        let input = arr2(&[[0.5, 0.3]]);
        let _ = net.forward(input.view()); // populate caches
        let loss_grad = Array2::ones((1, 1));
        let grads = net.backward(loss_grad);
        assert_eq!(grads.len(), net.layers.len());
    }

    #[test]
    fn backward_gradient_shapes_match_layer_weights() {
        let mut net = tiny_net();
        let input = arr2(&[[0.5, 0.3]]);
        let _ = net.forward(input.view());
        let loss_grad = Array2::ones((1, 1));
        let grads = net.backward(loss_grad);
        for (layer, grad) in net.layers.iter().zip(grads.iter()) {
            assert_eq!(grad.weights.dim(), layer.weights.dim());
            assert_eq!(grad.bias.len(), layer.bias.len());
        }
    }

    /// Numeric gradient check (finite differences) for a softmax-output network
    /// trained with categorical cross-entropy. This is the regression test for
    /// the double-counted softmax derivative: the CE gradient already accounts
    /// for the softmax derivative, so the analytic backward must match a pure
    /// numeric differentiation of the forward pass.
    #[test]
    fn softmax_output_backward_matches_finite_difference() {
        let mut net = Network::new()
            .add_layer(Layer::new(
                2,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::Xavier,
            ))
            .add_layer(Layer::new(
                4,
                4,
                ActivationFunction::Sigmoid,
                WeightInitializer::Xavier,
            ))
            .add_layer(Layer::new(
                4,
                2,
                ActivationFunction::Softmax,
                WeightInitializer::Xavier,
            ))
            .build();

        let x = arr2(&[[0.3, -0.7], [0.9, 0.2], [-0.4, 0.8]]);
        let y = arr1(&[1.0, 0.0, 1.0]);

        let pred = net.forward(x.view());
        let n_samples = pred.nrows() as f64;
        let upstream = LossFunction::CategoricalCrossEntropy.gradient(pred.view(), y.view());
        let upstream = upstream / n_samples;
        let grads = net.backward(upstream);

        let eps = 1e-6;
        let loss_of = |net: &mut Network| -> f64 {
            let p = net.forward(x.view());
            LossFunction::CategoricalCrossEntropy
                .compute(p.view(), y.view())
                .mean()
                .unwrap()
        };

        for layer_idx in 0..net.layers.len() {
            let weights_dim = net.layers[layer_idx].weights.dim();
            for i in 0..weights_dim.0 {
                for j in 0..weights_dim.1 {
                    let old = net.layers[layer_idx].weights[[i, j]];
                    net.layers[layer_idx].weights[[i, j]] = old + eps;
                    let l_plus = loss_of(&mut net);
                    net.layers[layer_idx].weights[[i, j]] = old - eps;
                    let l_minus = loss_of(&mut net);
                    net.layers[layer_idx].weights[[i, j]] = old;

                    let numeric = (l_plus - l_minus) / (2.0 * eps);
                    let analytic = grads[layer_idx].weights[[i, j]];
                    let diff = (analytic - numeric).abs();
                    assert!(
                        diff < 1e-5,
                        "layer {layer_idx} w[{i},{j}]: analytic={analytic:.8} numeric={numeric:.8} diff={diff:.8}"
                    );
                }
            }
        }
    }
}
