import numpy as np

class IOModel:
    """
    Core Input-Output Model.
    Phase 1 (Calibration): Initialize with the technical coefficient matrix (A) and value added coefficients (VA_coeffs).
    Phase 2 (Simulation): Stateless methods to compute new states given demand shocks.
    """
    def __init__(self, A: np.ndarray, VA_coeffs: np.ndarray = None):
        """
        Calibrate the IO Model from technical coefficients A.
        """
        self.n = A.shape[0]
        self.A = A.astype(float)
        
        if VA_coeffs is not None:
            self.VA_coeffs = VA_coeffs.astype(float)
        else:
            self.VA_coeffs = 1.0 - self.A.sum(axis=0)
            
        # Calculate Leontief Inverse: (I - A)^-1
        try:
            self.L = np.linalg.inv(np.eye(self.n) - self.A)
        except np.linalg.LinAlgError:
            self.L = np.eye(self.n) # Fallback if singular

    def simulate(self, new_Y: np.ndarray) -> np.ndarray:
        """
        Phase 2: Simulate new total output X given a new final demand vector Y.
        
        Returns:
            np.ndarray: New Total Output (X)
        """
        return self.L @ new_Y

    def simulate_shock(self, delta_Y: np.ndarray) -> np.ndarray:
        """
        Phase 2: Simulate change in total output given a shock to final demand.
        
        Returns:
            np.ndarray: Delta Total Output (X)
        """
        return self.L @ delta_Y


