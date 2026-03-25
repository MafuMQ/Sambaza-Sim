import pytest
import os
import sys

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def test_imports_and_db_models():
    """Ensure DB models and layout components can be imported without error."""
    from db.repositories.good_repo import GoodsDatabase
    from ui.layout import app
    assert app is not None
    
    # Check that database gives us a valid class
    db = GoodsDatabase()
    assert db is not None

def test_build_io_matrix():
    """Test building the IO matrix."""
    from core.io_matrix import build_io_matrix
    # demoDB=False assumes we have the data.db from the repo
    # If the database is missing, this might fail, but it expects a built DB.
    A, VA, isic_map = build_io_matrix(demoDB=False)
    
    assert A is not None
    assert VA is not None
    assert isinstance(isic_map, dict)
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == len(isic_map)

def test_run_simulation():
    """Test running a simple simulation."""
    from core.simulation import run_simulation
    
    res = run_simulation(
        uniform_demand=1000.0,
        iterations=1
    )
    
    assert 'before' in res
    assert 'after' in res
    assert 'deltas' in res
    assert 'isic_map' in res
