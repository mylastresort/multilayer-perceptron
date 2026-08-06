use mlp::training::loss::LossFunction;
use ndarray::{arr1, arr2};

#[test]
fn loss_function_parse_accepts_common_spellings() {
    assert_eq!(
        LossFunction::parse("binary_cross_entropy"),
        Some(LossFunction::BinaryCrossEntropy)
    );
    assert_eq!(
        LossFunction::parse("binaryCrossentropy"),
        Some(LossFunction::BinaryCrossEntropy)
    );
    assert_eq!(
        LossFunction::parse("binary_crossentropy"),
        Some(LossFunction::BinaryCrossEntropy)
    );
    assert_eq!(
        LossFunction::parse("bce"),
        Some(LossFunction::BinaryCrossEntropy)
    );
    assert_eq!(
        LossFunction::parse("categorical_cross_entropy"),
        Some(LossFunction::CategoricalCrossEntropy)
    );
    assert_eq!(
        LossFunction::parse("categoricalCrossentropy"),
        Some(LossFunction::CategoricalCrossEntropy)
    );
    assert_eq!(
        LossFunction::parse("categorical_crossentropy"),
        Some(LossFunction::CategoricalCrossEntropy)
    );
    assert_eq!(
        LossFunction::parse("cce"),
        Some(LossFunction::CategoricalCrossEntropy)
    );
    assert_eq!(LossFunction::parse("bogus"), None);
}

#[test]
fn loss_function_serde_round_trips() {
    let loss = LossFunction::BinaryCrossEntropy;
    let json = serde_json::to_string(&loss).unwrap();
    let back: LossFunction = serde_json::from_str(&json).unwrap();
    assert_eq!(loss, back, "serde round-trip failed for {json}");
    assert_eq!(LossFunction::default(), LossFunction::BinaryCrossEntropy);

    let categorical = LossFunction::CategoricalCrossEntropy;
    let json = serde_json::to_string(&categorical).unwrap();
    let back: LossFunction = serde_json::from_str(&json).unwrap();
    assert_eq!(categorical, back, "serde round-trip failed for {json}");
}

#[test]
fn cce_compute_is_negative_log_likelihood_of_true_class() {
    let preds = arr2(&[[0.7, 0.3], [0.1, 0.9]]);
    let targets = arr1(&[0.0, 1.0]);
    let losses = LossFunction::CategoricalCrossEntropy.compute(preds.view(), targets.view());
    assert!((losses[0] + 0.7f64.ln()).abs() < 1e-12);
    assert!((losses[1] + 0.9f64.ln()).abs() < 1e-12);
}

#[test]
fn cce_compute_matches_bce_for_two_class_softmax() {
    let preds = arr2(&[[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]]);
    let targets = arr1(&[1.0, 0.0, 1.0]);
    let bce = LossFunction::BinaryCrossEntropy.compute(preds.view(), targets.view());
    let cce = LossFunction::CategoricalCrossEntropy.compute(preds.view(), targets.view());
    for (b, c) in bce.iter().zip(cce.iter()) {
        assert!((b - c).abs() < 1e-12, "BCE {b} != CCE {c}");
    }
}

#[test]
fn cce_gradient_shape_matches_predictions() {
    let preds = arr2(&[[0.7, 0.3], [0.1, 0.9]]);
    let targets = arr1(&[0.0, 1.0]);
    let grad = LossFunction::CategoricalCrossEntropy.gradient(preds.view(), targets.view());
    assert_eq!(grad.dim(), preds.dim());
}

#[test]
fn cce_gradient_is_minus_onehot_over_prediction() {
    let preds = arr2(&[[0.7, 0.3], [0.1, 0.9]]);
    let targets = arr1(&[0.0, 1.0]);
    let grad = LossFunction::CategoricalCrossEntropy.gradient(preds.view(), targets.view());
    assert!((grad[[0, 0]] + 1.0 / 0.7).abs() < 1e-12);
    assert!(grad[[0, 1]].abs() < 1e-12);
    assert!(grad[[1, 0]].abs() < 1e-12);
    assert!((grad[[1, 1]] + 1.0 / 0.9).abs() < 1e-12);
}

#[test]
#[should_panic]
fn cce_gradient_panics_on_out_of_range_target() {
    let preds = arr2(&[[0.5, 0.5]]);
    let targets = arr1(&[2.0]);
    LossFunction::CategoricalCrossEntropy.gradient(preds.view(), targets.view());
}

#[test]
fn bce_compute_finite_for_valid_predictions() {
    let preds = arr2(&[[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]);
    let targets = arr1(&[1.0, 0.0]);
    let losses = LossFunction::BinaryCrossEntropy.compute(preds.view(), targets.view());
    assert!(losses.iter().all(|l| l.is_finite()));
    assert!(losses.iter().all(|l| *l > 0.0));
}

#[test]
fn bce_gradient_shape_matches_predictions() {
    let preds = arr2(&[[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]);
    let targets = arr1(&[1.0, 0.0]);
    let grad = LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
    assert_eq!(grad.dim(), preds.dim());
}

#[test]
fn bce_gradient_is_derivative_wrt_positive_class_output() {
    let preds = arr2(&[[0.2, 0.5, 0.3]]);
    let targets = arr1(&[1.0]);
    let grad = LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
    assert!((grad[[0, 2]] - (0.3 - 1.0) / (0.3 * 0.7)).abs() < 1e-12);
    assert!(grad[[0, 0]].abs() < 1e-12);
    assert!(grad[[0, 1]].abs() < 1e-12);
}

#[test]
fn bce_single_output_compute_and_gradient() {
    let preds = arr2(&[[0.8], [0.3]]);
    let targets = arr1(&[1.0, 0.0]);
    let loss = LossFunction::BinaryCrossEntropy.compute(preds.view(), targets.view());
    assert!((loss[0] + 0.8f64.ln()).abs() < 1e-12);
    assert!((loss[1] + 0.7f64.ln()).abs() < 1e-12);

    let grad = LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
    assert!((grad[[0, 0]] - (0.8 - 1.0) / (0.8 * 0.2)).abs() < 1e-12);
    assert!((grad[[1, 0]] - (0.3 - 0.0) / (0.3 * 0.7)).abs() < 1e-12);
}

#[test]
fn bce_binary_output_uses_subject_formula() {
    let preds = arr2(&[[0.7, 0.3], [0.1, 0.9]]);
    let targets = arr1(&[1.0, 0.0]);
    let loss = LossFunction::BinaryCrossEntropy.compute(preds.view(), targets.view());
    assert!((loss[0] + 0.3f64.ln()).abs() < 1e-12);
    assert!((loss[1] + 0.1f64.ln()).abs() < 1e-12);

    let grad = LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
    assert!((grad[[0, 1]] - (0.3 - 1.0) / (0.3 * 0.7)).abs() < 1e-12);
    assert!(grad[[0, 0]].abs() < 1e-12);
    assert!((grad[[1, 1]] - 0.9 / (0.9 * 0.1)).abs() < 1e-12);
    assert!(grad[[1, 0]].abs() < 1e-12);
}

#[test]
#[should_panic]
fn bce_gradient_panics_on_row_mismatch() {
    let preds = arr2(&[[0.5, 0.5]]);
    let targets = arr1(&[0.0, 1.0]);
    LossFunction::BinaryCrossEntropy.gradient(preds.view(), targets.view());
}
