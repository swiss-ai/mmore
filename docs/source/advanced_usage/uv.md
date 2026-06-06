# ⚡ `uv`
Use `uv` to install `mmore`

## Overview

`uv` is an extremely fast Python package and project manager, written in Rust. 

It can be used as a replacement or wrapper around `pip` to speed up the installation and environment management.


<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/astral-sh/uv/assets/1309177/03aa9163-1c79-4a87-a31d-7a9311ed9310">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/astral-sh/uv/assets/1309177/629e59c0-9c6e-4013-9ad4-adb2bcf5080d">
    <img alt="Shows a bar chart with benchmark results." src="https://github.com/astral-sh/uv/assets/1309177/629e59c0-9c6e-4013-9ad4-adb2bcf5080d">
  </picture>
</p>

## 📦 Install `uv`

Install `uv` with the standalone installers.

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 🛠️ Install `mmore` with `uv`
First create a virtual environment at the repository location:
```bash
uv venv
source .venv/bin/activate
```

Then install `mmore` by using `uv` with the usual `pip` command:

```bash
uv pip install -e .
```


## Notes

Using `uv` is an alternative to a standard `pip`-based installation.

It can be useful when you want faster dependency installation or a more streamlined environment setup.

## See also

- [Installation](../getting_started/installation.md)
