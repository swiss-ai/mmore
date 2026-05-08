# Developer Documentation

Welcome to the MMORE developer documentation! This guide will help you set up your development environment and contribute to the project.

## Table of Contents

- [Developer Documentation](#developer-documentation)
  - [Table of Contents](#table-of-contents)
  - [Development Setup](#development-setup)
    - [System Dependencies](#system-dependencies)
      - [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
      - [MacOS](#macos)
    - [Installing MMORE for Development](#installing-mmore-for-development)
    - [Code Quality-Tools](#code-quality-tools)
      - [Pre-commit Hooks](#pre-commit-hooks)
      - [Type Checking](#type-checking)
  - [Contributing Guidelines](#contributing-guidelines)
    - [Reporting Issues](#reporting-issues)
    - [Code Contributions](#code-contributions)
  - [Project Structure](#project-structure)
  - [Testing](#testing)
    - [Running tests in the terminal](#running-tests-in-the-terminal)
    - [Writing tests](#writing-tests)
  - [Pull Request Process](#pull-request-process)
    - [PR Checklist](#pr-checklist)
  - [Development Tips](#development-tips)
    - [Working with UV](#working-with-uv)
  - [Questions?](#questions)

---

## Development Setup

### System Dependencies

Before installing MMORE for development, ensure you have the required system dependencies installed.

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y ffmpeg libsm6 libxext6 libnss3 \
  libxi6 libxrandr2 libxcomposite1 libxcursor1 libxdamage1 \
  libxext6 libxfixes3 libxrender1 libasound2 libatk1.0-0 libgtk-3-0 libreoffice \
  libpango-1.0-0 libpangoft2-1.0-0 weasyprint
```

> **Note:** Note: On Ubuntu 24.04, replace `libasound2` with `libasound2t64`. You may also need to add the repository for Ubuntu 20.04 focal to have access to a few of the sources (e.g., create `/etc/apt/sources.list.d/mmore.list` with the contents `deb http://cz.archive.ubuntu.com/ubuntu focal main universe`).

#### MacOS

```bash
brew update
brew install ffmpeg gtk+3 pango cairo \
  gobject-introspection libffi pkg-config libx11 libxi \
  libxrandr libxcomposite libxcursor libxdamage libxext \
  libxrender atk libreoffice weasyprint
```

If `weasyprint` fails to find GTK or Cairo, also run:

```bash
brew install cairo pango gdk-pixbuf libffi
uv pip install weasyprint
```

### Installing MMORE for Development

**1. Clone the repository:**

```bash
git clone https://github.com/swiss-ai/mmore.git
cd mmore
```

**2. Create a virtual environment and install dependencies:**

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[all,cpu,dev]"
```

> **GPU (CUDA 12.6):** replace `cpu` with `cu126` - e.g. `uv pip install -e ".[all,cu126,dev]"`
>
> **Partial install:** replace `all` with only the stages you need - e.g. `uv pip install -e ".[rag,cpu,dev]"` for RAG only. Available stages: `process`, `index`, `rag`, `api`.

> **Important:** This package requires many big dependencies and requires a dependency override, so it must be installed with `uv` to handle `pip` installations. Check our [tutorial on uv](./uv.md) for more information.

### Code Quality-Tools

MMORE uses several tools to maintain code quality and consistency.

#### Pre-commit Hooks

We use `pre-commit` to automatically run code formatters and linters before each commit.

**Setup**

**1. Install pre-commit** (if not already installed):

```bash
uv pip install pre-commit
```

**2. Set up the git hook scripts:**

```bash
pre-commit install
```

**3. Run the checks manually** (optional but recommended before your first commit):

```bash
pre-commit run --all-files
```

**Configured Hooks**

The pre-commit configuration runs `ruff`, a code formatter for consistent style

#### Type Checking

We use pyright for static type checking. Please ensure your Pull Requests are type-checked.

To run type checking manually:

```bash
pyright
```

## Contributing Guidelines

We welcome contributions! Here's how you can help:

### Reporting Issues

- **Bug Reports:** Open an issue with a clear description, steps to reproduce, and expected vs. actual behavior
- **Feature Requests:** Open an issue describing the feature, its use case, and potential implementation approach
- Check the [Issues](https://github.com/swiss-ai/mmore/issues) page for ongoing work

### Code Contributions

1. **Fork the repository** and create a new branch for your feature/fix
2. **Write clear, documented code** following the existing style
3. **Add tests** if applicable
4. **Ensure all pre-commit hooks pass**
5. **Run type checking** with `pyright`
6. **Submit a Pull Request** with a clear description

## Project Structure

mmore/
├── mmore/
│   ├── process/          # Document processing pipeline
│   │   ├── processors/   # Individual file type processors
│   │   └── ...
│   ├── postprocess/      # Post-processing utilities
│   ├── index/            # Indexing and vector DB
│   ├── rag/              # RAG implementation
│   └── type/             # Type definitions and data models
├── docs/                 # Documentation
├── examples/             # Example configurations and data
├── tests/                # Test suite
├── .pre-commit-config.yaml
├── pyproject.toml
└── README.md

Key Modules
- **`mmore.process`**: Handles extraction from various file formats
- **`mmore.index`**: Manages hybrid dense+sparse indexing with Milvus
- **`mmore.rag`**: RAG system with LangChain integration
- **`mmore.type`**: Core data structures like `MultimodalSample`

## Testing

### Running tests in the terminal

```bash
pytest tests/
```

### GPU tests and the `--gpu` flag

A few tests require a CUDA GPU (e.g. `test_rerank` in `tests/test_rag.py` which
loads `bge-reranker-base`, and the end-to-end pipeline test in
`tests/test_full_pipeline_gpu.py`). They are marked with `@pytest.mark.gpu` and
**skipped by default** so the suite stays runnable on a laptop or in CPU-only
CI.

To run them, pass the `--gpu` flag (registered in `tests/conftest.py`):

```bash
pytest --gpu                                  # full suite, including GPU tests
pytest --gpu -m gpu                           # only the GPU-marked tests
pytest --gpu tests/test_full_pipeline_gpu.py  # one specific GPU test file
```

Without `--gpu`, a `pytest_collection_modifyitems` hook adds a
`pytest.mark.skip(reason="Pass --gpu to run GPU tests")` to every test carrying
the `gpu` marker — they appear as **skipped**, not failed.

`--gpu` is a declaration that a CUDA device is actually available. If CUDA
isn't present despite the flag, the GPU tests will fail loudly (`RuntimeError`)
rather than silently skip — that's intentional, so a misconfigured CI
environment doesn't go unnoticed.

#### Running on RCP (RunAI / Kubernetes)

The repo doesn't have a dedicated GPU-tests workflow yet. Adapt the pattern in
[rcp_and_production.md](./rcp_and_production.md) for an interactive job:

```bash
runai submit mmore-tests-gpu \
  --image ghcr.io/swiss-ai/mmore:student-uid<UID>-gid<GID>-gpu \
  --node-pool h100 \
  --pvc light-scratch:/lightscratch \
  --gpu 1 \
  --run-as-gid <GID> \
  --preemptible \
  --attach --interactive --tty \
  --command /bin/bash
```

Then inside the container:

```bash
cd /workspace/mmore
uv pip install -e ".[process,index,rag,api,cu126,dev,websearch]"
export HF_HOME=/lightscratch/users/$USER/.cache/huggingface
pytest --gpu -m gpu
```

For a one-shot non-interactive job, replace
`--attach --interactive --tty --command /bin/bash` with:

```
--command "bash -lc 'cd /workspace/mmore && pytest --gpu -m gpu'"
```

#### Running on CSCS (SLURM)

Standard SLURM pattern with a venv (or a singularity/sarus container):

```bash
# job_tests_gpu.sbatch
#!/bin/bash
#SBATCH --job-name=mmore-tests-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=tests-%j.log

module load <cuda-modules>          # cluster-specific
source $SCRATCH/venvs/mmore/bin/activate
export HF_HOME=$SCRATCH/.cache/huggingface
cd $SCRATCH/mmore
pytest --gpu -m gpu
```

Submit with `sbatch job_tests_gpu.sbatch`. For interactive iteration on a
GPU node:

```bash
srun --gres=gpu:1 --pty --time=01:00:00 bash
source $SCRATCH/venvs/mmore/bin/activate
pytest --gpu
```

#### Marking a new GPU-only test

When you add a test that needs a GPU, mark it explicitly:

```python
import pytest

@pytest.mark.gpu
def test_something_on_gpu():
    ...
```

The `gpu` marker is registered in `pyproject.toml` under
`[tool.pytest.ini_options].markers`, so no `PytestUnknownMarkWarning` is
emitted on collection.

#### Cheat sheet

| Target | Command |
|---|---|
| Local CPU, all non-GPU | `pytest` |
| Local CPU, single file | `pytest tests/test_retriever.py` |
| Local GPU, all | `pytest --gpu` |
| Local GPU, GPU-only | `pytest --gpu -m gpu` |
| Stop on first failure, verbose | `pytest -xv [...]` |
| RCP interactive | `runai submit ... --attach --interactive`, then `pytest --gpu -m gpu` |
| CSCS one-shot | `sbatch job_tests_gpu.sbatch` |

### Writing tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Cover edge cases and error conditions
- Mock external dependencies when appropriate
- Mark GPU-only tests with `@pytest.mark.gpu` (see above)

## Pull Request Process

1. **Update documentation** if you're adding new features
2. **Add examples** for new functionality
3. **Ensure all tests pass** and pre-commit hooks succeed
4. **Update the changelog** if applicable
5. **Request review** from maintainers

### PR Checklist

- [] Code follows project style guidelines
- [] Pre-commit hooks pass (`pre-commit run --all-files`)
- [] Type checking passes (`pyright`)
- [] Tests added/updated as needed
- [] Documentation updated
- [] Examples provided for new features
- [] Commit messages are clear and descriptive

## Development Tips

### Working with UV

- Use `uv pip` instead of `pip` for all package installations
- The project uses dependency overrides that are handled automatically by `uv`
- See the UV tutorial for more details

## Questions?

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Reach out to the maintainers
- Check existing issues for similar questions

Thank you for contributing to MMORE! 🎉