import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, time


def load_params(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_orders(path: str, order_date: str = "2024-01-15") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["order_time"] = pd.to_datetime(order_date + " " + df["order_time"])
    df["inventory_available"] = df["inventory_available"].astype(bool)
    return df


def load_customers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def load_trucks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cold_capable"] = df["cold_capable"].astype(int)
    df["can_use_after_16"] = df["can_use_after_16"].astype(int)
    return df


def load_productivity(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_all(data_dir: str, config_path: str, order_date: str = "2024-01-15") -> dict:
    data_dir = Path(data_dir)
    return {
        "params": load_params(config_path),
        "orders": load_orders(data_dir / "sample_orders.csv", order_date),
        "customers": load_customers(data_dir / "sample_customers.csv"),
        "trucks": load_trucks(data_dir / "sample_trucks.csv"),
        "productivity": load_productivity(data_dir / "sample_productivity.csv"),
    }
