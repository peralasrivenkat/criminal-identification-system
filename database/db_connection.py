from __future__ import annotations

import mysql.connector
from mysql.connector import Error

from config import ensure_directories
from database.db_config import DB_CONFIG


def create_server_connection():
    ensure_directories()
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        port=DB_CONFIG["port"],
    )


def create_connection():
    ensure_directories()
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        port=DB_CONFIG["port"],
    )


def close_connection(connection) -> None:
    if connection is not None and connection.is_connected():
        connection.close()
