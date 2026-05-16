import pandas as pd
from datetime import timedelta

CATEGORY_MULTIPLIERS = {
    "FMCG":     {"pick": 1.00, "pack": 1.00, "load": 1.00},
    "Fresh":    {"pick": 1.20, "pack": 1.10, "load": 1.00},
    "Cold":     {"pick": 1.30, "pack": 1.20, "load": 1.10},
    "Cosmetic": {"pick": 1.10, "pack": 1.50, "load": 1.00},
    "Bulky":    {"pick": 1.40, "pack": 1.30, "load": 1.40},
}


def calculate_order_prepare_time(order: pd.Series, params: dict) -> float:
    """Return preparation time in minutes."""
    pick_mult = 1.20 if order.get("fresh_flag", 0) else 1.00
    pack_mult = 1.10 if order.get("fresh_flag", 0) else 1.00
    load_mult = 1.00

    if order.get("cold_flag", 0):
        pick_mult = max(pick_mult, 1.30)
        pack_mult = max(pack_mult, 1.20)
        load_mult = 1.10

    base = params["base_prepare_sec"]
    pick = order["sku_count"] * params["pick_sec_per_sku"] * pick_mult
    pack = order["carton_qty"] * params["pack_sec_per_carton"] * pack_mult
    load = order["carton_qty"] * params["load_sec_per_carton"] * load_mult
    cold_buf = params["cold_buffer_sec"] if order.get("cold_flag", 0) else 0
    fresh_buf = params["fresh_buffer_sec"] if order.get("fresh_flag", 0) else 0

    total_sec = base + pick + pack + load + cold_buf + fresh_buf
    return total_sec / 60.0  # return minutes


def add_ready_time(orders: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add prepare_min and ready_time columns."""
    orders = orders.copy()
    orders["prepare_min"] = orders.apply(
        lambda row: calculate_order_prepare_time(row, params), axis=1
    )
    orders["ready_time"] = orders["order_time"] + pd.to_timedelta(orders["prepare_min"], unit="m")
    return orders
