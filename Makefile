.PHONY: build run test check fmt clippy clean

build:
	cargo build

run:
	cargo run

test:
	cargo test

check:
	cargo check

fmt:
	cargo fmt --all

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

clean:
	cargo clean
