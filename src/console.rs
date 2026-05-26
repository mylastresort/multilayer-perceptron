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
