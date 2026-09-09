"""Build a standalone aibench executable with PyInstaller.

    pip install -e ".[build]"      # or: pip install pyinstaller
    python build_exe.py

Produces dist/aibench (dist/aibench.exe on Windows) — a single self-contained
binary with the Python runtime and all dependencies bundled. No Python install
is needed on the target machine. Build per-platform (a Windows build runs on
Windows, a Linux build on Linux).
"""

import shutil
import sys

import PyInstaller.__main__


def main() -> None:
    PyInstaller.__main__.run([
        "pyi_entry.py",
        "--name", "aibench",
        "--onefile",
        "--console",
        "--clean",
        "--noconfirm",
        # aibench imports some submodules indirectly (e.g. the scoring registry);
        # collect them all so nothing is missed by static analysis.
        "--collect-submodules", "aibench",
    ])
    exe = "aibench.exe" if sys.platform.startswith("win") else "aibench"
    print(f"\nBuilt dist/{exe}")
    if shutil.which("upx") is None:
        print("(tip: install UPX to shrink the binary further)")


if __name__ == "__main__":
    main()
