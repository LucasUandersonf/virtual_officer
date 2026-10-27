"""Offline research lab: multi-asset walk-forward search for a robust spot setup.

Runs as a separate process (``python -m mqmas.research``) so the CPU-heavy
grid never blocks the trading event loop. Nothing here places orders.
"""
