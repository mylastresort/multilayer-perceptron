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
	cargo clippy --lib --bins --all-features -- -W clippy::too_many_lines -D warnings

clean:
	cargo clean
	rm -rf reports/ models/model.json
	rm -f data/train.csv data/val.csv data/data_training.csv data/data_test.csv
