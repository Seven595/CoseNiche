# Refactoring Guide

This guide describes how to make tutorial scripts more consistent and easier to maintain.

## Goals

1. Remove hard-coded paths.
2. Use configuration files and command-line arguments.
3. Move shared logic into `tutorials/utils/`.
4. Split scripts into small functions.
5. Add clear logging and error handling.

## Shared Utilities

```
tutorials/utils/
├── io.py
├── gene_filtering.py
├── visualization.py
└── logger.py
```

## Recommended Pattern

```python
import argparse
from pathlib import Path
from tutorials.utils import load_config, setup_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir

    logger = setup_logger(
        name="analysis",
        log_file=Path(config["output"]["output_dir"]) / "run.log",
    )

    logger.info("Starting analysis")
    # Analysis code goes here.
    logger.info("Analysis completed")


if __name__ == "__main__":
    main()
```

## Checklist

- [ ] Remove hard-coded paths.
- [ ] Add argument parsing.
- [ ] Load settings from YAML.
- [ ] Use helpers from `tutorials.utils`.
- [ ] Use `logger` instead of `print` where practical.
- [ ] Add useful error messages.
- [ ] Add docstrings.
- [ ] Update the relevant README.
- [ ] Run the script on a small example.

## Priorities

High priority:
- Refactor deconvolution scripts.
- Update configuration files.
- Keep data paths outside code.

Medium priority:
- Refactor attention analysis scripts.
- Refactor spatial communication scripts.
- Unify visualization style.

Low priority:
- Add unit tests for utilities.
- Add integration tests for main scripts.
- Optimize memory and parallel processing.
