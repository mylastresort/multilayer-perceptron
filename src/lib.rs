//! # Multilayer Perceptron
//!
//! A from-scratch MLP implementation in Rust for binary/multi-class classification.
//! No ML framework — every operation (forward, backward, optimizer, loss) is hand-written
//! using `ndarray` for linear algebra.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mlp::network::{model::Network, layer::Layer, activation::ActivationFunction, initializer::WeightInitializer};
//! use mlp::training::{loss::LossFunction, optimizer::OptimizerType};
//!
//! let mut net = Network::new()
//!     .add_layer(Layer::new(30, 24, ActivationFunction::Sigmoid, WeightInitializer::He))
//!     .add_layer(Layer::new(24, 24, ActivationFunction::Sigmoid, WeightInitializer::He))
//!     .add_layer(Layer::new(24, 2, ActivationFunction::Softmax, WeightInitializer::He))
//!     .learning_rate(0.03)
//!     .build();
//! ```

pub mod console;
pub mod data;
pub mod network;
pub mod training;
pub mod visualization;
