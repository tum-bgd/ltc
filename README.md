[![PyPI - Version](https://img.shields.io/pypi/v/ltcpy)](https://pypi.org/project/ltcpy)
[![Crates.io Version](https://img.shields.io/crates/v/ltc-rs)](https://crates.io/crates/ltc-rs)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/tum-bgd/ltc/publish.yml)](https://github.com/tum-bgd/ltc/actions/workflows/publish.yml)

# Lifetime Clustering (LTC)

Implementation of the Lifetime Clustering (LTC) algorithm, as described in [Efficient and robust topology-based clustering](https://doi.org/10.1016/j.patcog.2026.114415).

<p align="center">
<img src="https://raw.githubusercontent.com/tum-bgd/ltc/refs/heads/main/assets/graphical_abstract.jpg"/>
<figcaption>Graphical abstract of the LTC algorithm (zollner2026efficient, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>).</figcaption>
</p>

## Example

This is a minimal example. See [installation](#installation) for how to install.

### Python

```python
from ltcpy import LifetimeClustering

data = ... # data
eps = ... # radius

ltc = LifetimeClustering(eps) # initialize
ltc.fit(data) # cluster

labels = ltc.labels_ # get labels
```
Check our [demo notebook](https://github.com/tum-bgd/ltc/blob/main/scripts/demo.ipynb) for more elaborated examples.

### Rust

To use directly in Rust, add it with `cargo add ltc-rs` as a dependency to your `Cargo.toml`.

```rust
let x: Vec<Vec<f32>> = ...; // data
let eps: f32 = ...; // radius

let (labels, lifetime) = ltc_rs::fit(&x,eps); // cluster
```

## Installation

Install with `pip` from [`PyPI`](https://pypi.org/project/ltcpy/) with:

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
@article{zollner2026efficient,
title = {Efficient and robust topology-based clustering},
author = {Johann Maximilian Zollner and Balthasar Teuscher and Wejdene Mansour and Martin Werner},
journal = {Pattern Recognition},
volume = {180},
pages = {114415},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.114415},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326013804}
}
```

## Funding

This work was supported by the German Federal Ministry of Research, Technology and Space under grant number 16DKWN134.

## License

The project is licensed under the [Apache-2.0 license](https://github.com/tum-bgd/ltc/blob/main/LICENSE) or [opensource.org/licenses/Apache-2.0](https://opensource.org/licenses/Apache-2.0).
