import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple


def check_route_capacity(route_orders: pd.DataFrame, truck: pd.Series) -> bool:
    return (
        route_orders["weight_kg"].sum() <= truck["max_kg"]
        and route_orders["cbm"].sum() <= truck["max_cbm"]
    )


def check_cold_compatibility(route_orders: pd.DataFrame, truck: pd.Series) -> bool:
    if route_orders["cold_flag"].sum() > 0:
        return truck["cold_capable"] == 1
    return True


def check_truck_access(customer: pd.Series, truck: pd.Series) -> bool:
    access = customer.get("truck_access_type", "5T_allowed")
    truck_type = truck["truck_type"]
    if access == "2T_only" and truck_type not in ["2T", "cold_2T", "exception_van"]:
        return False
    if access == "cold_required" and truck["cold_capable"] != 1:
        return False
    # small_vehicle_only: exception_van or cold_2T (small enough to access narrow lanes)
    if access == "small_vehicle_only" and truck_type not in ["exception_van", "cold_2T"]:
        return False
    return True


def simulate_route_timing(
    route_customers: List[dict],
    departure_time: datetime,
    travel_matrix: pd.DataFrame,
    fc_id: str,
    avg_speed_kmh: float = 25.0
) -> Tuple[List[dict], datetime]:
    """
    Simulate arrival times for a route starting at departure_time from FC.
    Returns list of stop dicts with arrival_time, and the route finish time.
    """
    from src.travel_time import estimate_travel_time_min

    stops = []
    current_time = departure_time

    for i, cust in enumerate(route_customers):
        if i == 0:
            # from FC to first customer
            row = travel_matrix[
                (travel_matrix["from_id"] == fc_id) &
                (travel_matrix["to_id"] == cust["customer_id"])
            ]
            if len(row) > 0:
                travel_min = row.iloc[0]["travel_time_min"]
            else:
                travel_min = 30.0
        else:
            prev = route_customers[i - 1]
            row = travel_matrix[
                (travel_matrix["from_id"] == prev["customer_id"]) &
                (travel_matrix["to_id"] == cust["customer_id"])
            ]
            if len(row) > 0:
                travel_min = row.iloc[0]["travel_time_min"]
            else:
                travel_min = 30.0

        arrival = current_time + timedelta(minutes=travel_min)

        # wait if arrive before receive_start
        receive_start = datetime.strptime(
            arrival.strftime("%Y-%m-%d") + " " + cust["receive_start"], "%Y-%m-%d %H:%M"
        )
        if arrival < receive_start:
            arrival = receive_start

        service_end = arrival + timedelta(minutes=cust["unload_min"])
        stops.append({
            "customer_id": cust["customer_id"],
            "arrival_time": arrival,
            "service_end": service_end,
        })
        current_time = service_end

    # Return to FC
    if route_customers:
        last = route_customers[-1]
        row = travel_matrix[
            (travel_matrix["from_id"] == last["customer_id"]) &
            (travel_matrix["to_id"] == fc_id)
        ]
        return_min = row.iloc[0]["travel_time_min"] if len(row) > 0 else 30.0
        finish_time = current_time + timedelta(minutes=return_min)
    else:
        finish_time = current_time

    return stops, finish_time


def build_greedy_routes(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    trucks: pd.DataFrame,
    travel_matrix: pd.DataFrame,
    params: dict,
    order_date: str = "2024-01-15"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feasible routes using greedy heuristic.
    Returns (route_summary_df, stop_plan_df).

    Algorithm:
    1. Filter same-day eligible orders
    2. Calculate departure time = max(route_start_after, max(ready_time))
    3. Group by zone, sort by priority/cold/distance
    4. Assign cold orders first to cold trucks
    5. Fill routes respecting capacity, time windows, and operational deadline
    6. Mark unassigned orders with reason
    """
    eligible = orders[orders["same_day_eligible"] == True].copy()
    customers_dict = customers.set_index("customer_id").to_dict("index")

    route_start_dt = pd.to_datetime(f"{order_date} {params['route_start_after']}")
    operational_deadline = pd.to_datetime(f"{order_date} {params['operational_deadline']}")
    fc_id = params["fc_id"]
    avg_speed = params["avg_speed_kmh"]

    # Merge customer info into orders
    eligible = eligible.merge(
        customers[["customer_id", "lat", "lng", "zone", "receive_start", "receive_end",
                   "unload_min", "truck_access_type", "priority"]],
        on="customer_id", how="left", suffixes=("", "_cust")
    )

    # Sort: cold first, then vip, then by zone
    eligible["sort_key"] = (
        eligible["cold_flag"].astype(int) * -100
        + eligible["fresh_flag"].astype(int) * -50
        + (eligible["priority"] == "vip").astype(int) * -10
    )
    eligible = eligible.sort_values("sort_key")

    # Track which orders are assigned
    assigned_order_ids = set()
    unassigned_reasons = {}
    routes = []
    stop_plan_rows = []
    route_id_counter = 1

    # Separate cold and normal trucks
    cold_trucks = trucks[trucks["cold_capable"] == 1].copy()
    normal_trucks = trucks[trucks["cold_capable"] == 0].copy()

    def get_deadline(truck_row, order_date, params):
        if truck_row["can_use_after_16"] == 1:
            return pd.to_datetime(f"{order_date} {params['customer_promise_deadline']}")
        return pd.to_datetime(f"{order_date} {params['operational_deadline']}")

    def try_assign_route(candidate_orders, truck_row, order_date):
        """Try to form a feasible route from candidate orders using given truck."""
        if len(candidate_orders) == 0:
            return None, []

        deadline = get_deadline(truck_row, order_date, params)

        route_orders = []
        route_customers = []
        seen_customers = set()
        total_kg = 0
        total_cbm = 0

        for _, order in candidate_orders.iterrows():
            if order["order_id"] in assigned_order_ids:
                continue
            cust_id = order["customer_id"]
            if cust_id not in customers_dict:
                continue
            cust = customers_dict[cust_id]

            # check truck access
            if not check_truck_access(pd.Series(cust), truck_row):
                continue

            # check capacity
            if total_kg + order["weight_kg"] > truck_row["max_kg"]:
                continue
            if total_cbm + order["cbm"] > truck_row["max_cbm"]:
                continue

            # check cold compatibility
            if order["cold_flag"] == 1 and truck_row["cold_capable"] != 1:
                continue

            route_orders.append(order)
            if cust_id not in seen_customers:
                seen_customers.add(cust_id)
                route_customers.append({
                    "customer_id": cust_id,
                    "receive_start": cust["receive_start"],
                    "receive_end": cust["receive_end"],
                    "unload_min": cust["unload_min"],
                })
            total_kg += order["weight_kg"]
            total_cbm += order["cbm"]

            if len(route_customers) >= truck_row.get("max_stops", 30):
                break

        if not route_orders:
            return None, []

        # Calculate departure time
        max_ready = max(o["ready_time"] for o in route_orders)
        departure = max(route_start_dt, max_ready)

        # Simulate timing
        stops, finish_time = simulate_route_timing(
            route_customers, departure, travel_matrix, fc_id, avg_speed
        )

        # Check if finish before deadline
        if finish_time <= deadline:
            return (route_orders, route_customers, departure, stops, finish_time, total_kg, total_cbm), route_orders

        # Try splitting: take fewer customers
        for n in range(len(route_customers) - 1, 0, -1):
            sub_customers = route_customers[:n]
            sub_cust_ids = {c["customer_id"] for c in sub_customers}
            sub_orders = [o for o in route_orders if o["customer_id"] in sub_cust_ids]
            if not sub_orders:
                continue
            max_ready2 = max(o["ready_time"] for o in sub_orders)
            departure2 = max(route_start_dt, max_ready2)
            sub_kg = sum(o["weight_kg"] for o in sub_orders)
            sub_cbm = sum(o["cbm"] for o in sub_orders)
            stops2, finish2 = simulate_route_timing(
                sub_customers, departure2, travel_matrix, fc_id, avg_speed
            )
            if finish2 <= deadline:
                return (sub_orders, sub_customers, departure2, stops2, finish2,
                        sub_kg, sub_cbm), sub_orders

        return None, []

    def assign_batch(batch_orders, preferred_trucks):
        nonlocal route_id_counter
        remaining = batch_orders.copy()

        for _, truck_row in preferred_trucks.iterrows():
            if remaining.empty:
                break
            unassigned_batch = remaining[~remaining["order_id"].isin(assigned_order_ids)]
            if unassigned_batch.empty:
                break

            # Keep trying this truck until no more orders can be assigned to it
            while True:
                unassigned_batch = remaining[~remaining["order_id"].isin(assigned_order_ids)]
                if unassigned_batch.empty:
                    break

                result, assigned = try_assign_route(unassigned_batch, truck_row, order_date)
                if result is None or not assigned:
                    break

                route_orders_list, route_customers, departure, stops, finish_time, total_kg, total_cbm = result

                route_id = f"R{route_id_counter:03d}"
                route_id_counter += 1
                order_ids = [o["order_id"] for o in route_orders_list]
                assigned_order_ids.update(order_ids)

                deadline = get_deadline(truck_row, order_date, params)
                sla_ok = finish_time <= deadline

                # Compute route distance
                route_km = 0.0
                prev_id = fc_id
                for c in route_customers:
                    row = travel_matrix[
                        (travel_matrix["from_id"] == prev_id) &
                        (travel_matrix["to_id"] == c["customer_id"])
                    ]
                    if len(row) > 0:
                        route_km += row.iloc[0]["distance_km"]
                    prev_id = c["customer_id"]
                # return leg
                row = travel_matrix[
                    (travel_matrix["from_id"] == prev_id) &
                    (travel_matrix["to_id"] == fc_id)
                ]
                if len(row) > 0:
                    route_km += row.iloc[0]["distance_km"]

                routes.append({
                    "route_id": route_id,
                    "truck_id": truck_row["truck_id"],
                    "truck_type": truck_row["truck_type"],
                    "zone": batch_orders["zone"].iloc[0] if "zone" in batch_orders.columns else "unknown",
                    "num_customers": len(route_customers),
                    "num_orders": len(order_ids),
                    "depart_time": departure.strftime("%H:%M"),
                    "finish_time": finish_time.strftime("%H:%M"),
                    "finish_datetime": finish_time,
                    "total_kg": round(total_kg, 1),
                    "total_cbm": round(total_cbm, 2),
                    "kg_utilization": round(total_kg / truck_row["max_kg"], 3),
                    "cbm_utilization": round(total_cbm / truck_row["max_cbm"], 3),
                    "route_km": round(route_km, 1),
                    "sla_status": "OK" if sla_ok else "LATE",
                    "deadline": deadline.strftime("%H:%M"),
                    "order_ids": order_ids,
                    "fixed_cost": truck_row["fixed_cost"],
                    "cost_per_km": truck_row["cost_per_km"],
                    "driver_cost_per_hour": truck_row["driver_cost_per_hour"],
                    "cold_capable": truck_row["cold_capable"],
                    "can_use_after_16": truck_row["can_use_after_16"],
                })

                for stop in stops:
                    cust_id = stop["customer_id"]
                    cust_orders = [o for o in route_orders_list if o["customer_id"] == cust_id]
                    for o in cust_orders:
                        arr = stop["arrival_time"]
                        deadline_dt = get_deadline(truck_row, order_date, params)
                        stop_plan_rows.append({
                            "route_id": route_id,
                            "truck_id": truck_row["truck_id"],
                            "order_id": o["order_id"],
                            "customer_id": cust_id,
                            "zone": batch_orders["zone"].iloc[0] if "zone" in batch_orders.columns else "unknown",
                            "arrival_time": arr,
                            "deadline": deadline_dt,
                            "on_time": arr <= deadline_dt,
                            "weight_kg": o["weight_kg"],
                            "cbm": o["cbm"],
                        })

                remaining = remaining[~remaining["order_id"].isin(assigned_order_ids)]

    # Zone-based greedy assignment
    zones = eligible["zone"].dropna().unique()

    for zone in zones:
        zone_orders = eligible[eligible["zone"] == zone].copy()

        # Cold orders in this zone
        cold_zone = zone_orders[zone_orders["cold_flag"] == 1]
        normal_zone = zone_orders[zone_orders["cold_flag"] == 0]

        # Assign cold orders to cold trucks first
        if not cold_zone.empty and not cold_trucks.empty:
            assign_batch(cold_zone, cold_trucks)
            # Mark still-unassigned cold orders
            still_cold = cold_zone[~cold_zone["order_id"].isin(assigned_order_ids)]
            for oid in still_cold["order_id"]:
                unassigned_reasons[oid] = "no_cold_truck_available"

        # Assign normal orders to non-cold trucks first, then try cold trucks for stragglers
        non_cold_trucks = trucks[trucks["cold_capable"] == 0]
        if not normal_zone.empty and not non_cold_trucks.empty:
            assign_batch(normal_zone, non_cold_trucks)

        # Retry unassigned normal orders with cold trucks (e.g. small_vehicle_only locations)
        remaining_normal = normal_zone[~normal_zone["order_id"].isin(assigned_order_ids)]
        if not remaining_normal.empty and not cold_trucks.empty:
            assign_batch(remaining_normal, cold_trucks)

    # Mark unassigned eligible orders
    all_unassigned = eligible[~eligible["order_id"].isin(assigned_order_ids)]
    for _, row in all_unassigned.iterrows():
        if row["order_id"] not in unassigned_reasons:
            unassigned_reasons[row["order_id"]] = "no_feasible_route"

    route_df = pd.DataFrame(routes) if routes else pd.DataFrame()
    stop_df = pd.DataFrame(stop_plan_rows) if stop_plan_rows else pd.DataFrame()

    if route_df.empty:
        route_df = pd.DataFrame(columns=[
            "route_id", "truck_id", "truck_type", "zone", "num_customers", "num_orders",
            "depart_time", "finish_time", "total_kg", "total_cbm", "kg_utilization",
            "cbm_utilization", "route_km", "sla_status", "deadline", "order_ids",
            "fixed_cost", "cost_per_km", "driver_cost_per_hour", "cold_capable", "can_use_after_16"
        ])

    route_df.attrs["unassigned_reasons"] = unassigned_reasons
    stop_df.attrs["unassigned_reasons"] = unassigned_reasons

    return route_df, stop_df
