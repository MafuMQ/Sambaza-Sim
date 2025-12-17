from __future__ import annotations
import pandas as pd

class FinalDemand:

    C:list[float] = [] #Household Consumption Expenditure | Expenditures by households on goods and services, including durable goods, nondurable goods, and services.
    I:list[float] = [] #Gross Fixed Capital Formation | Business investments in fixed assets (e.g., machinery, buildings) plus residential construction.
    G:list[float] = [] #Government Consumption Expenditure | Government spending on goods and services for public use.
    X:list[float] = [] #Exports of Goods and Services | Value of all goods and services sold to other countries.
    M:list[float] = [] #Imports of Goods and Services | Value of all goods and services purchased from other countries.
    FD:list[float] = [] #Total Final Demand | Sum of all final demand components (C + I + G + X - M).

    def __init__(self, C: list[float] = None, I: list[float] = None, G: list[float] = None, X: list[float] = None, M: list[float] = None, FD: list[float] = None, Evaluate_Final_Demand: bool = True) -> None: # pyright: ignore[reportArgumentType]
        self.C = C
        self.I = I
        self.G = G
        self.X = X
        self.M = M
        self.FD = FD
        self.evaluate_final_demand() if Evaluate_Final_Demand else None

    @classmethod
    def load_from_csv(cls, file_path: str = "io1FD.csv") -> FinalDemand:
        df = pd.read_csv(file_path)
        return cls(
            C=df["C"].tolist(),
            I=df["I"].tolist(),
            G=df["G"].tolist(),
            X=df["X"].tolist(),
            M=df["M"].tolist(),
            FD=df["FD"].tolist()
        )

    def evaluate_final_demand(self) -> None:
        """
        Evaluates the final demand for each component and updates the Total_Final_Demand (FD).
        """
        self.FD = [
            c + i + g + x - m for c, i, g, x, m in zip(self.C, self.I, self.G, self.X, self.M)
        ]
