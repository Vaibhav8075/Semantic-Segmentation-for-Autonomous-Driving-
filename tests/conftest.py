import os
import sys

# Make the project modules importable when running `pytest` from any directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
