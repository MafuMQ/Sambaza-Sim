import os
import sys

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ui.layout import app
from ui import callbacks  # registers callbacks as side-effect

if __name__ == '__main__':
    app.run(debug=True, port=8050)
