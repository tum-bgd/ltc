# Lifetime Clustering (LTC)

Efficient and robust topology-based clustering.

## Example

This is a minimal example. Use `cargo add ltc-rs` to add it as a dependency to `Cargo.toml`.

```rust
let x: Vec<Vec<f32>> = ...; // data
let eps: f32 = ...; // fixed-radius

let (labels, lifetime) = ltc_rs::fit(&x,eps); // cluster
```

## License

Licensed under either of

 * Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license (http://opensource.org/licenses/MIT)

at your option.