from Good_Indice import *
from Production import *
from enum import Enum


class InvestmentType(Enum):
    PRODUCTION = "production"
    INDICE = "indice"

class Investment:

    def __init__(self, id: int, name: str, type_of_investment: InvestmentType, produce_isic: str, implementation_cost: float):
        self.id = id
        self.name = name
        if isinstance(type_of_investment, InvestmentType):
            self.type = type_of_investment
        else:
            raise ValueError("type_of_investment must be an instance of InvestmentType Enum")
        self.produce_isic = produce_isic
        self.implementation_cost = implementation_cost
        self.produce_id = None

        self.production_inputs = None  # Inputs required for production in JSON format
        self.production_added_values = None  # Value added for the good
        self.total_inputs_cost = None  # Total input price used in production
        self.total_value_added = None  # Total value added by the production method
        self.price = None  # Price of the good
        self.price_history = None  # Historical price data in JSON format
        self.quantity = None  # Quantity of the good

    def __repr__(self):
        return f"Investment(id={self.id}, name={self.name}, type={self.type.value}, produce_id={self.produce_id}, produce_isic={self.produce_isic}, implementation_cost={self.implementation_cost})"
    
    def set_investment_metrics(self, inputs, added_values):
        self.production_inputs = inputs
        self.production_added_values = added_values
        self.total_inputs_cost = sum(inputs.values())
        self.total_value_added = sum(added_values.values())
        self.price = self.total_inputs_cost + self.total_value_added

    def apply_investment(self):
        if self.type == InvestmentType.PRODUCTION:
            self.apply_investment_on_production()
        elif self.type == InvestmentType.INDICE:
            self.apply_investment_on_indicie()
        else:
            raise ValueError(f"Unknown investment type: {self.type}")

    def apply_investment_on_production(self):
        ptdb = ProductionsDatabase()
        pass

    def apply_investment_on_indicie(self, use_demo_db: bool = True):
        gidb = GoodsIndiceDatabase(database_url="sqlite:///dataDEMO.db") if use_demo_db else GoodsIndiceDatabase()
        gidb.update_good_indice_by_isic(self.produce_isic,
                                production_inputs=self.production_inputs, 
                                production_added_values=self.production_added_values,
                                total_inputs_cost=self.total_inputs_cost, 
                                total_value_added=self.total_value_added, 
                                price=self.price)