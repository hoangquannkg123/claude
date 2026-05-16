import numpy as np
import pandas as pd


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlng / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def estimate_travel_time_min(lat1, lng1, lat2, lng2, avg_speed_kmh: float = 25.0) -> float:
    """Estimate travel time in minutes using straight-line distance with road factor."""
    dist_km = haversine_km(lat1, lng1, lat2, lng2) * 1.4  # road factor
    return (dist_km / avg_speed_kmh) * 60.0


def build_travel_matrix(customers: pd.DataFrame, fc_lat: float, fc_lng: float,
                         fc_id: str, avg_speed_kmh: float = 25.0) -> pd.DataFrame:
    """Build full travel time matrix between FC and all customers."""
    rows = []
    locations = pd.concat([
        pd.DataFrame([{
            "customer_id": fc_id,
            "lat": fc_lat,
            "lng": fc_lng
        }]),
        customers[["customer_id", "lat", "lng"]]
    ], ignore_index=True)

    for _, src in locations.iterrows():
        for _, dst in locations.iterrows():
            if src["customer_id"] == dst["customer_id"]:
                continue
            dist = haversine_km(src["lat"], src["lng"], dst["lat"], dst["lng"]) * 1.4
            travel = estimate_travel_time_min(src["lat"], src["lng"], dst["lat"], dst["lng"], avg_speed_kmh)
            rows.append({
                "from_id": src["customer_id"],
                "to_id": dst["customer_id"],
                "distance_km": round(dist, 2),
                "travel_time_min": round(travel, 1),
            })
    return pd.DataFrame(rows)
