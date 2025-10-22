import random
from datetime import datetime
from typing import Dict

class MarketConnector:
    """Simulates connection to ISO/RTO (e.g. CAISO, ERCOT)"""
    
    def __init__(self, region="CAISO"):
        self.region = region
    
    def fetch_realtime_prices(self) -> Dict[str, float]:
        """Fetch LMP (Locational Marginal Price)"""
        # Mock API call
        base_price = 45.0 # $/MWh
        
        hour = datetime.utcnow().hour
        if 8 <= hour <= 20: # On-peak
            lmp = base_price * random.uniform(1.2, 3.0)
        else:
            lmp = base_price * random.uniform(0.5, 0.9)
            
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "region": self.region,
            "lmp_price": round(lmp, 2),
            "currency": "USD",
            "unit": "MWh"
        }
