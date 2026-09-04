# No Python in this project

This project is **pure Rust**. No Python code anywhere — no test data generators, no helpers, no build scripts.

## Rules
- No `.py` files anywhere in the repo
- No `python` or `pip` dependencies
- Test data generators are Rust binaries (e.g. `generate_test_wavs`)
- All tooling uses `cargo`
