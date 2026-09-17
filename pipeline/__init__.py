"""
Data pipeline package
======================

Handles data ingestion between raw CSV files and the running application:
- ``setup_data``        — Seeds the database from CSV files or random data
- ``load_tech_changes`` — Loads CSV scenario configs into DB and rebuilds EXAMPLES dict
"""


