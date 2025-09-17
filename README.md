# OptiDiff-Mat

Neural Diffusion Models for Optical-Property Guided Materials Discovery

## Installation

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/Kasperjoergensen3/OptiDiff-Mat.git
cd OptiDiff-Mat
pip install -e .
```

### For Development

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

### For Documentation

Install with documentation dependencies:

```bash
pip install -e ".[docs]"
```

## Package Structure

```
OptiDiff-Mat/
├── pyproject.toml          # Package configuration and dependencies
├── setup.py                # Backward compatibility setup script
├── README.md               # This file
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore patterns
├── src/                   # Source code
│   └── optidiff_mat/      # Main package
│       ├── __init__.py    # Package initialization
│       ├── models/        # Neural diffusion models
│       ├── data/          # Data loading and preprocessing
│       └── utils/         # Utility functions
└── tests/                 # Test suite
    ├── __init__.py
    └── test_basic.py
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
