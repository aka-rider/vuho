.PHONY: all build package run clean test lint

all: build

build:
	cargo build --release -p vuho-ui

package: build
	./scripts/package.sh

run: package
	open Vuho.app

clean:
	cargo clean

test:
	cargo test

lint:
	cargo clippy --workspace --all-targets
	@echo "--- cargo deny (advisory-only, non-blocking) ---"
	-cargo deny check
