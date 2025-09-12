import os
import pymysql
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

def get_connection():
    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=int(os.getenv("DB_PORT", 3306)),
            cursorclass=pymysql.cursors.DictCursor  # optional: return results as dictionaries
        )
        print("Connected to MySQL successfully.")
        return conn
    except Exception as e:
        print("Failed to connect to MySQL:", e)
        raise
