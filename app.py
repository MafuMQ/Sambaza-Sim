import os
import sys
from pathlib import Path

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Bootstrap DB before importing the UI (layout.py queries the DB at import time)
db_file = Path(root_dir) / 'data.db'
if not db_file.exists():
    print("Database not found. Setting up data...")
    from pipeline.setup_data import setup_data
    setup_data(source="data/ex2", overwrite_existing_data=True)
    print("Data setup complete.\n")

from ui.layout import app
from ui import callbacks  # registers callbacks as side-effect

if __name__ == '__main__':
    app.run(debug=True, port=8050)
