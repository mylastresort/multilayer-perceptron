use mlp::app::split::{SplitArgs, run_split};

#[test]
fn run_split_rejects_ratio_of_zero() {
    let result = run_split(&SplitArgs {
        dataset_path: format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR")),
        train_out: "/tmp/mlp_split_train_dummy.csv".to_string(),
        val_out: "/tmp/mlp_split_val_dummy.csv".to_string(),
        ratio: 0.0,
    });
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("(0, 1)"), "unexpected error: {msg}");
}

#[test]
fn run_split_rejects_ratio_of_one() {
    let result = run_split(&SplitArgs {
        dataset_path: format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR")),
        train_out: "/tmp/mlp_split_train_dummy.csv".to_string(),
        val_out: "/tmp/mlp_split_val_dummy.csv".to_string(),
        ratio: 1.0,
    });
    assert!(result.is_err());
}

#[test]
fn run_split_creates_train_and_val_csv_files() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let train_path = format!("/tmp/mlp_split_train_{}_{}.csv", std::process::id(), ts);
    let val_path = format!("/tmp/mlp_split_val_{}_{}.csv", std::process::id(), ts);

    let result = run_split(&SplitArgs {
        dataset_path: format!("{}/data/data.csv", env!("CARGO_MANIFEST_DIR")),
        train_out: train_path.clone(),
        val_out: val_path.clone(),
        ratio: 0.8,
    });

    let header_cols = std::fs::read_to_string(&train_path)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .split(',')
        .count();
    let data_cols = std::fs::read_to_string(&train_path)
        .unwrap()
        .lines()
        .nth(1)
        .unwrap()
        .split(',')
        .count();
    assert_eq!(
        header_cols, data_cols,
        "train.csv header/data column mismatch"
    );

    let _ = std::fs::remove_file(&train_path);
    let _ = std::fs::remove_file(&val_path);

    assert!(result.is_ok(), "run_split failed: {:?}", result.err());
}

#[test]
fn run_split_rejects_dataset_with_fewer_than_two_rows() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let csv_path = format!("/tmp/mlp_tiny_{}_{}.csv", std::process::id(), ts);
    std::fs::write(&csv_path, "id,diagnosis,f1\n1,M,0.5\n").unwrap();
    let result = run_split(&SplitArgs {
        dataset_path: csv_path.clone(),
        train_out: format!("/tmp/mlp_split_train_tiny_{ts}.csv"),
        val_out: format!("/tmp/mlp_split_val_tiny_{ts}.csv"),
        ratio: 0.7,
    });
    let _ = std::fs::remove_file(&csv_path);
    let Err(e) = result else {
        panic!("expected error for < 2 rows")
    };
    assert!(e.to_string().contains("at least 2 rows"), "unexpected: {e}");
}
