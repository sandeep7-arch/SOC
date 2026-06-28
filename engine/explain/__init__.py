import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPLAIN_DIR = os.path.join(PROJECT_ROOT, "Engine_Assistant", "Explain")

if EXPLAIN_DIR not in __path__:
    __path__.append(EXPLAIN_DIR)
