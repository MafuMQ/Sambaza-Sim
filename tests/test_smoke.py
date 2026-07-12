import pytest
import os
import sys

# Ensure root path is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def test_imports_and_db_models():
    """Ensure DB models and layout components can be imported without error."""
    from pipeline.db.repositories.good_repo import GoodsDatabase
    from presentation.layout import app
    assert app is not None
    
    # Check that database gives us a valid class
    db = GoodsDatabase()
    assert db is not None

def test_build_io_matrix():
    # 1) Rebuild A and VA with the new data
    from pipeline.builders.level1_matrix import build_io_matrix
    
    # 2) Actually get them. Assumes we have the data.db from the repo
    # If the database is missing, this might fail, but it expects a built DB.
    A, VA, isic_map = build_io_matrix(demoDB=False)
    
    assert A is not None
    assert VA is not None
    assert isinstance(isic_map, dict)
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == len(isic_map)

def test_simulate_shock():
    """Test running a simple simulation on the stateless IOModel."""
    from pipeline.builders.level1_matrix import build_io_matrix
    from core.io_matrix import IOModel
    import numpy as np
    
    A, VA, isic_map = build_io_matrix(demoDB=False)
    model = IOModel(A=A, VA_coeffs=VA)
    
    base_demand = np.full(model.n, 1000.0)
    X = model.simulate(base_demand)
    
    assert X is not None
    assert X.shape == (model.n,)


