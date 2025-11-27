[![PyPI - Version](https://img.shields.io/pypi/v/ltcpy)](https://pypi.org/project/ltcpy)
[![Crates.io Version](https://img.shields.io/crates/v/ltc-rs)](https://crates.io/crates/ltc-rs)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/tum-bgd/ltc/publish.yml)](https://github.com/tum-bgd/ltc/actions/workflows/publish.yml)

# Lifetime Clustering (LTC)

Efficient and robust topology-based clustering.

<p align="center">
<img src="https://raw.githubusercontent.com/tum-bgd/ltc/refs/heads/main/assets/graphical_abstract.jpg"/>
<figcaption>Graphical abstract of the LTC algorithm (JM Zollner, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>).</figcaption>
</p>

## Example

This is a minimal example. See [installation](#installation) for how to install.

### Python

```python
from ltcpy import LifetimeClustering

data = ... # data
eps = ... # fixed-radius

ltc = LifetimeClustering(eps) # initialize
ltc.fit(data) # cluster

labels = ltc.labels_ # get labels
```
### Rust

To use directly in Rust, add it with `cargo add ltc-rs` as a dependency to your `Cargo.toml`.

```rust
let x: Vec<Vec<f32>> = ...; // data
let eps: f32 = ...; // fixed-radius

let (labels, lifetime) = ltc_rs::fit(&x,eps); // cluster
```

## Installation

Installing with with `pip` from [`PyPI`](https://pypi.org/project/ltcpy/) with

```bash
pip install ltcpy
```

Alternatively, build from source with [Rust](https://rust-lang.org/tools/install) and [Maturin](https://www.maturin.rs/installation.html). To build and install the `ltcpy` package locally, run:

```sh
maturin develop -m ltc-py/Cargo.toml --release
```

## Layout

| Path        | Content                                        |
| ----------- | ---------------------------------------------- |
| `./assets`  | Example data                                   |
| `./ltc-py`  | Python bindings                                |
| `./ltc-rs`  | Rust implementation                            |
| `./scripts` | Notebook with examples, Python implementation |

## Reference

```bibtex
@article{ltc,
    author    = {Zollner, Johann M. and Teuscher, Balthasar and Mansour, Wejdene and Werner, Martin},
    title     = {Efficient and Robust Topology-Based Clustering},
}
```

## Funding

This work was supported by the German Federal Ministry of Research, Technology and Space under grant number 16DKWN134.

## License

The project is licensed under the [Apache-2.0 license](https://github.com/tum-bgd/ltc/blob/main/LICENSE) or [opensource.org/licenses/Apache-2.0](https://opensource.org/licenses/Apache-2.0).
