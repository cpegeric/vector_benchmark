"""
Database Utilities

This module contains shared functions for database connectivity and configuration.

Functions:
- `get_db_connection`: Establishes a connection to the database using `pymysql`.
- `set_env`: Reads environment variables from the configuration file (`cfg.json`) and applies them to the current database session using `SET variable = value` queries.
  This allows for tuning experimental features (like HNSW index parameters) without modifying Python code.
"""
import pymysql

def get_db_connection(config, use_db=True):
    return pymysql.connect(
        host=config['host'],
        port=config.get('port', 6001),
        user=config['user'],
        password=config['password'],
        database=config['database'] if use_db else None,
        autocommit=True
    )

def set_env(cursor, config):
    env = config.get('env', {})
    for key, value in env.items():
        sql = f"SET {key} = {value}"
        # print(f"Executing: {sql}")
        cursor.execute(sql)
