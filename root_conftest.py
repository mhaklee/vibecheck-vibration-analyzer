import sys
import os

# Make the project root importable so pytest can find the vibecheck package
# without requiring a full pip install in CI.
sys.path.insert(0, os.path.dirname(__file__))
