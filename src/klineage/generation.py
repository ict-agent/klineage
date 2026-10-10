"""Generate kernels with ``python -m klineage.generation PROBLEM_DIR``."""

from klineage.cli.generation import generate, main

__all__ = ["generate", "main"]


if __name__ == "__main__":
    main()
