.PHONY: all re build run test check fmt clippy clean doc open-doc

all: build

re: clean build

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

doc:
	PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig cargo doc --no-deps --open
