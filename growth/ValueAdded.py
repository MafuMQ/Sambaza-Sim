from __future__ import annotations
import pandas as pd

class ValueAdded:

    Wages:list[float] = [] #Compensation of Employees (Wages & Salaries) | Includes wages, salaries, benefits, and social contributions paid to workers.
    Taxes:list[float] = [] #Taxes less Subsidies on Production | Indirect taxes (e.g., VAT, payroll taxes) minus government subsidies to firms.
    Surplus:list[float] = [] #Gross Operating Surplus (GOS) | Profits, interest, rent, and depreciation (capital consumption). This is the return to capital.
    Mixed_Income:list[float] = [] #Mixed Income (in some tables) | For unincorporated businesses (e.g., self-employed), where it's hard to separate labor and capital income.
    Value_Added:list[float] = [] #Total Value Added | Sum of all value-added components (Wages + Taxes + Surplus + Mixed Income).

    def __init__(self, Wages: list[float] = None, Taxes: list[float] = None, Surplus: list[float] = None, Mixed_Income: list[float] = None, Total_Value_Added: list[float] = None, Evaluate_Total_Value_Added:bool = True) -> None: # pyright: ignore[reportArgumentType]
        self.Wages = Wages
        self.Taxes = Taxes
        self.Surplus = Surplus
        self.Mixed_Income = Mixed_Income
        self.Value_Added = Total_Value_Added
        self.evaluate_value_added() if Evaluate_Total_Value_Added else None

    @classmethod
    def load_from_csv(cls, file_path: str = "io1VA.csv") -> ValueAdded:
        df = pd.read_csv(file_path, index_col=0)
        return cls(
            Wages=df.loc["Wages"].values.tolist(),
            Taxes=df.loc["Taxes"].values.tolist(),
            Surplus=df.loc["Surplus"].values.tolist(),
            Mixed_Income=df.loc["Mixed_Income"].values.tolist(),
            Total_Value_Added=df.loc["Value_Added"].values.tolist()
        )

    def evaluate_value_added(self) -> None:
        """
        Evaluates the value added for each component and updates the Total_Value_Added.
        """
        self.Value_Added = [ 
            w + t + s + m for w, t, s, m in zip(self.Wages, self.Taxes, self.Surplus, self.Mixed_Income)
        ]


    def breakdown_by_percentage(self) -> dict:
        """
        Breaks down the total value added by percentage for each component.
        """
        total = sum(self.Value_Added) if self.Value_Added else 1
        return {
            "Wages": sum(self.Wages) / total * 100,
            "Taxes": sum(self.Taxes) / total * 100,
            "Surplus": sum(self.Surplus) / total * 100,
            "Mixed Income": sum(self.Mixed_Income) / total * 100,
        }
