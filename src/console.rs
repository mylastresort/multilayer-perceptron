#[derive(Clone, Copy)]
pub enum Tone {
    Info,
    Success,
    Warn,
    Accent,
    Muted,
    TrainMetric,
    ValMetric,
}

fn colors_enabled() -> bool {
    std::env::var("NO_COLOR").is_err()
}

fn code(tone: Tone) -> &'static str {
    match tone {
        Tone::Info => "34",        // Blue
        Tone::Success => "32",     // Green
        Tone::Warn => "33",        // Yellow
        Tone::Accent => "36",      // Cyan
        Tone::Muted => "90",       // Bright black / gray
        Tone::TrainMetric => "32", // Green
        Tone::ValMetric => "35",   // Magenta
    }
}

pub fn paint(value: &str, tone: Tone) -> String {
    if !colors_enabled() {
        return value.to_string();
    }
    format!("\x1b[{}m{}\x1b[0m", code(tone), value)
}

pub fn bold(value: &str) -> String {
    if !colors_enabled() {
        return value.to_string();
    }
    format!("\x1b[1m{}\x1b[0m", value)
}

#[cfg(test)]
mod tests {
    use super::{bold, paint, Tone};
    use std::sync::Mutex;

    // All tests that touch NO_COLOR must hold this lock to prevent races.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn paint_with_no_color_returns_plain_string() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("NO_COLOR", "1"); }
        let result = paint("hello", Tone::Info);
        unsafe { std::env::remove_var("NO_COLOR"); }
        assert_eq!(result, "hello");
    }

    #[test]
    fn bold_with_no_color_returns_plain_string() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("NO_COLOR", "1"); }
        let result = bold("world");
        unsafe { std::env::remove_var("NO_COLOR"); }
        assert_eq!(result, "world");
    }

    #[test]
    fn paint_without_no_color_adds_ansi_codes() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("NO_COLOR"); }
        let result = paint("hi", Tone::Success);
        assert!(result.contains("hi"));
        assert!(result.contains('\x1b'));
    }

    #[test]
    fn bold_without_no_color_adds_ansi_codes() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("NO_COLOR"); }
        let result = bold("hi");
        assert!(result.contains("hi"));
        assert!(result.contains('\x1b'));
    }

    #[test]
    fn paint_with_warn_tone_adds_ansi_codes() {
        let _g = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("NO_COLOR"); }
        let result = paint("warning!", Tone::Warn);
        assert!(result.contains("warning!"));
        assert!(result.contains('\x1b'));
    }
}
