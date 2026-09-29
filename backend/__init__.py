"""Hindsight backend package.

Loading .env here guarantees the environment is ready no matter which module
is imported first, and no matter which folder uvicorn is started from.
"""
from pathlib import Path

from dotenv import load_dotenv

# utf-8-sig tolerates the BOM that Windows Notepad sometimes adds to .env files.
# Real environment variables always win over values in the file (override=False).
load_dotenv(Path(__file__).resolve().parent / ".env", encoding="utf-8-sig", override=False)