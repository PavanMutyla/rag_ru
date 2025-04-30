from enum import IntEnum, StrEnum, Enum

class KBCategory(str, Enum):
    ProductCategory = "product_category"
    InvestmentRegulations = "investment_regulations"
    TaxationDetails = "taxation_details"
    MarketSegments = "market_segments"
    CulturalAspects = "cultural_aspects"
    General = "general"

class THRESHOLD(Enum):
    threshold = 0.2