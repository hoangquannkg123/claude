import pandas as pd
from datetime import datetime


def calculate_route_cost(route: pd.Series, params: dict) -> dict:
    """Calculate cost components for one route."""
    fixed = route["fixed_cost"]

    dist_cost = route["route_km"] * route["cost_per_km"]

    # Duration in hours
    depart = pd.to_datetime("2024-01-15 " + route["depart_time"])
    finish = pd.to_datetime("2024-01-15 " + route["finish_time"])
    if finish < depart:
        finish = finish + pd.Timedelta(days=1)
    duration_h = (finish - depart).total_seconds() / 3600.0

    driver_cost = duration_h * route["driver_cost_per_hour"]

    cold_extra = 200000 if route.get("cold_capable", 0) == 1 else 0

    total = fixed + dist_cost + driver_cost + cold_extra
    return {
        "fixed_cost": fixed,
        "distance_cost": dist_cost,
        "driver_cost": driver_cost,
        "cold_extra": cold_extra,
        "route_cost": total,
        "route_duration_h": duration_h,
    }


def calculate_warehouse_cost(orders: pd.DataFrame, params: dict) -> dict:
    """Calculate warehouse labor cost."""
    if "same_day_eligible" in orders.columns:
        eligible = orders[orders["same_day_eligible"] == True]
    else:
        eligible = orders

    total_sku = eligible["sku_count"].sum()
    total_cartons = eligible["carton_qty"].sum()

    pick_sec = total_sku * params["pick_sec_per_sku"]
    pack_sec = total_cartons * params["pack_sec_per_carton"]
    load_sec = total_cartons * params["load_sec_per_carton"]

    pick_h = pick_sec / 3600.0
    pack_h = pack_sec / 3600.0
    load_h = load_sec / 3600.0

    pick_cost = pick_h * params["picker_cost_per_hour"]
    pack_cost = pack_h * params["packer_cost_per_hour"]
    load_cost = load_h * params["loader_cost_per_hour"]

    total = pick_cost + pack_cost + load_cost
    return {
        "pick_cost": pick_cost,
        "pack_cost": pack_cost,
        "load_cost": load_cost,
        "warehouse_cost": total,
        "total_pick_hours": pick_h,
        "total_pack_hours": pack_h,
        "loading_hours": load_h,
    }


def calculate_total_plan_cost(route_df: pd.DataFrame, orders: pd.DataFrame,
                               params: dict, late_orders: int = 0) -> dict:
    """Return total cost and breakdown."""
    if route_df.empty:
        route_costs = {"fixed_cost": 0, "distance_cost": 0, "driver_cost": 0,
                       "cold_extra": 0, "route_cost": 0}
    else:
        cost_rows = route_df.apply(lambda r: calculate_route_cost(r, params), axis=1)
        cost_df = pd.DataFrame(list(cost_rows))
        route_costs = cost_df.sum().to_dict()

    wh_cost = calculate_warehouse_cost(orders, params)
    penalty = late_orders * params["sla_penalty_per_late_order"]

    total = route_costs.get("route_cost", 0) + wh_cost["warehouse_cost"] + penalty

    return {
        "truck_fixed_cost": route_costs.get("fixed_cost", 0),
        "truck_distance_cost": route_costs.get("distance_cost", 0),
        "driver_time_cost": route_costs.get("driver_cost", 0),
        "cold_chain_cost": route_costs.get("cold_extra", 0),
        "warehouse_cost": wh_cost["warehouse_cost"],
        "sla_penalty": penalty,
        "total_cost": total,
        **wh_cost,
    }
