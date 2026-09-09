"""PyInstaller entry point for the standalone aibench executable.

Kept minimal: it just calls the CLI. The frozen-exe self-exec hook (used by the
Phase 2 code checks) lives in aibench.cli.main and runs before argparse.
"""

import sys

from aibench.cli import main

if __name__ == "__main__":
    sys.exit(main())
