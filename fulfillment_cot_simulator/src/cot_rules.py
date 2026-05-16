import pandas as pd
from datetime import datetime


def assign_cot_eligibility(orders: pd.DataFrame, params: dict, order_date: str = "2024-01-15") -> pd.DataFrame:
    """Add same_day_eligible flag based on order_time, COT, and inventory."""
    orders = orders.copy()
    cot_dt = pd.to_datetime(f"{order_date} {params['cot_time']}")
    orders["same_day_eligible"] = (
        (orders["order_time"] <= cot_dt) & (orders["inventory_available"] == True)
    )
    return orders


def get_effective_delivery_deadline(order_date: str, truck_row: pd.Series, params: dict) -> datetime:
    """Return 15:30 for normal trucks, customer promise for exception vehicles."""
    if truck_row["can_use_after_16"] == 1:
        return pd.to_datetime(f"{order_date} {params['customer_promise_deadline']}")
    else:
        return pd.to_datetime(f"{order_date} {params['operational_deadline']}")
