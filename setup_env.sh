#!/usr/bin/env bash
# Sets up a Python virtual environment for the Anisotropy notebooks.
# Usage:  bash setup_env.sh
set -u

cd "$(dirname "$0")" || exit 1
echo "==> Project: $(pwd)"

PKGS="ipykernel numpy sympy"

try_build () {
    local PY="$1"
    echo
    echo "==> Trying interpreter: $PY  ($("$PY" -V 2>&1))"
    rm -rf .venv
    "$PY" -m venv .venv || return 1
    # shellcheck disable=SC1091
    source .venv/bin/activate || return 1
    python -m pip install --quiet --upgrade pip || return 1
    echo "==> Installing: $PKGS"
    # shellcheck disable=SC2086
    python -m pip install $PKGS || return 1
    return 0
}

# Prefer a stable interpreter; fall back through what's installed.
CANDIDATES=""
for p in python3.13 python3.12 python3.11 python3; do
    command -v "$p" >/dev/null 2>&1 && CANDIDATES="$CANDIDATES $p"
done

if [ -z "$CANDIDATES" ]; then
    echo "!! No python3 found on PATH." >&2
    exit 1
fi
echo "==> Candidate interpreters:$CANDIDATES"

OK=0
for p in $CANDIDATES; do
    if try_build "$p"; then OK=1; break; fi
    echo "   ...failed with $p, trying next"
done

if [ "$OK" -ne 1 ]; then
    echo
    echo "!! Could not build an environment with any interpreter." >&2
    echo "!! Try:  brew install python@3.13   then re-run this script." >&2
    exit 1
fi

# Register the kernel so VS Code / Jupyter list it by name.
python -m ipykernel install --user \
    --name anisotropy --display-name "Python (Anisotropy)" >/dev/null 2>&1 \
    && echo "==> Registered Jupyter kernel: Python (Anisotropy)"

echo
echo "==> Installed versions:"
python - <<'PY'
import sys
print("   python :", sys.version.split()[0])
for m in ("numpy", "sympy", "ipykernel"):
    try:
        print(f"   {m:7s}:", __import__(m).__version__)
    except Exception as e:
        print(f"   {m:7s}: FAILED -> {e}")
PY

echo
echo "==> Smoke test of your notebook's imports:"
python -c "
import numpy as np
from sympy import *
x = symbols('x')
assert np.allclose(np.linalg.inv(np.eye(3)), np.eye(3))
assert integrate(x, x) == x**2/2
print('   numpy + sympy OK')
"

echo
echo "===================================================================="
echo " DONE. Interpreter path (copy this if VS Code asks):"
echo "   $(pwd)/.venv/bin/python"
echo
echo " In VS Code:"
echo "   1. Cmd+Shift+P -> 'Python: Select Interpreter' -> pick .venv"
echo "   2. Open main.ipynb, click the kernel name (top right),"
echo "      choose 'Python (Anisotropy)' or the .venv interpreter"
echo "===================================================================="
