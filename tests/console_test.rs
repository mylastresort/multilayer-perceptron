use mlp::console::{Tone, bold, paint};
use std::sync::Mutex;

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn paint_with_no_color_returns_plain_string() {
    let _g = ENV_LOCK.lock().unwrap();
    unsafe {
        std::env::set_var("NO_COLOR", "1");
    }
    let result = paint("hello", Tone::Info);
    unsafe {
        std::env::remove_var("NO_COLOR");
    }
    assert_eq!(result, "hello");
}

#[test]
fn bold_with_no_color_returns_plain_string() {
    let _g = ENV_LOCK.lock().unwrap();
    unsafe {
        std::env::set_var("NO_COLOR", "1");
    }
    let result = bold("world");
    unsafe {
        std::env::remove_var("NO_COLOR");
    }
    assert_eq!(result, "world");
}

#[test]
fn paint_without_no_color_adds_ansi_codes() {
    let _g = ENV_LOCK.lock().unwrap();
    unsafe {
        std::env::remove_var("NO_COLOR");
    }
    let result = paint("hi", Tone::Success);
    assert!(result.contains("hi"));
    assert!(result.contains('\x1b'));
}

#[test]
fn bold_without_no_color_adds_ansi_codes() {
    let _g = ENV_LOCK.lock().unwrap();
    unsafe {
        std::env::remove_var("NO_COLOR");
    }
    let result = bold("hi");
    assert!(result.contains("hi"));
    assert!(result.contains('\x1b'));
}

#[test]
fn paint_with_warn_tone_adds_ansi_codes() {
    let _g = ENV_LOCK.lock().unwrap();
    unsafe {
        std::env::remove_var("NO_COLOR");
    }
    let result = paint("warning!", Tone::Warn);
    assert!(result.contains("warning!"));
    assert!(result.contains('\x1b'));
}
