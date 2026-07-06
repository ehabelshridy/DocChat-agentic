"""
tests/conftest.py

Shared pytest configuration. Loads .env before any test so that
HF_TOKEN is available to tests that need it.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
