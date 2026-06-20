"""MongoDB connection and data fetching."""

import os
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

COLLECTION_NAME = "FM_AL_HIS_ANALYSIS"
DB_PREFIX = "shams_"


def get_client() -> MongoClient:
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGO_URI not set in .env file")
    return MongoClient(uri)


def list_plants(client: MongoClient) -> list[dict]:
    """Return list of {display_name, db_name} for every shams_ database."""
    dbs = client.list_database_names()
    plants = []
    for db in sorted(dbs):
        if db.startswith(DB_PREFIX):
            raw = db[len(DB_PREFIX):]
            display = raw.replace("_", " ").title()
            plants.append({"display_name": display, "db_name": db})
    return plants


def fetch_records(
    client: MongoClient,
    db_name: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Fetch all hourly records from FM_AL_HIS_ANALYSIS between start and end (inclusive)."""
    start_naive = start.replace(tzinfo=None)
    end_naive   = end.replace(hour=23, minute=59, second=59, tzinfo=None)
    cursor = client[db_name][COLLECTION_NAME].find(
        {"timestamp": {"$gte": start_naive, "$lte": end_naive}},
        sort=[("timestamp", 1)],
    )
    return list(cursor)


def fetch_temperature_records(
    client: MongoClient,
    db_name: str,
    start: datetime,
    end: datetime,
    collection_name: str = "FM_OD_PRD",
) -> list[dict]:
    """Fetch pv_temperature docs from FM_OD_PRD.

    Only queries the EMI device (Device Type='EMI') — it is the single
    environmental sensor that holds the real pv_temperature for the plant.
    FM_OD_PRD stores timestamp as a plain string "YYYY-MM-DD HH:MM:SS".
    """
    collection = client[db_name][collection_name]
    start_str = start.strftime("%Y-%m-%d 00:00:00")
    end_str = end.strftime("%Y-%m-%d 23:59:59")
    cursor = collection.find(
        {
            "timestamp": {"$gte": start_str, "$lte": end_str},
            "Device Type": "EMI",
        },
        projection={"timestamp": 1, "pv_temperature": 1, "_id": 0},
    )
    return list(cursor)
