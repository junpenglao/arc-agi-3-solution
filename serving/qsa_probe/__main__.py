"""Dispatch the explicit GPU-only QSA probe."""

from serving.qsa_probe.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
