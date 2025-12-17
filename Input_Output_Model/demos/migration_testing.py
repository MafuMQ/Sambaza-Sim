import numpy as np
import json

def load_supply_profile(json_data):
    tiers = json_data['tiers']
    tiers = sorted(tiers, key=lambda x: x['price'])
    
    # Pre-allocate arrays
    n = len(tiers)
    caps = np.zeros(n)
    prices = np.zeros(n)
    
    # Parse JSON into arrays
    for i, t in enumerate(tiers):
        # Handle the "Infinite" case for the last tier
        if t['cap'] == -1 or t['cap'] is None:
            caps[i] = np.inf 
        else:
            caps[i] = t['cap']
            
        prices[i] = t['price']
        
    # Create the "Lower Bounds" array
    # This is crucial for vectorization: it tells us where each tier starts.
    # We shift capacity by 1 to get cumulative sum, starting at 0.
    bounds = np.concatenate(([0], np.cumsum(caps)[:-1]))
    
    return bounds, caps, prices

# --- Example Usage ---
db_record = {
    "tiers": [{"cap": 10, "price": 5.0}, {"cap": 20, "price": 7.0}, {"cap": -1, "price": 12.0}]
}

bounds, caps, prices = load_supply_profile(db_record)
print("bounds =", bounds)
print("caps   =", caps)
print("prices =", prices)
# Result:
# bounds = [ 0.  10.  30.]
# caps   = [ 10.  20.  inf]
# prices = [ 5.   7.  12.]

def calculate_weighted_price_vectorized(demand, bounds, caps, prices):
    if demand <= 0: return 0.0

    # 1. Determine how much demand falls into each tier.
    # Logic: (Demand - Start of Tier), clipped between 0 and Tier Capacity
    amounts = np.clip(demand - bounds, 0, caps)
    
    # 2. Calculate total cost (Dot product)
    total_cost = np.dot(amounts, prices)
    
    # 3. Return average unit price
    return total_cost / demand

# --- Test ---
d = 11
avg_price = calculate_weighted_price_vectorized(d, bounds, caps, prices)

print(f"Demand: {d}")
print(f"Breakdown: {np.clip(d - bounds, 0, caps)}") # Shows [10. 15.  0.]
print(f"Unit Price: ${avg_price}")