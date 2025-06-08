import os
import pymysql
from fastapi import HTTPException
from fastapi import APIRouter, Query


router = APIRouter()


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        cursorclass=pymysql.cursors.DictCursor
    )


@router.get("/locations-in-bounds")
def get_locations_in_bounds(
    swLat: float = Query(...),
    swLng: float = Query(...),
    neLat: float = Query(...),
    neLng: float = Query(...),
    zoom: int = Query(...)
):
    # Decide which table to query based on zoom level
    table = None
    lat_col, lng_col = "latitude", "longitude"
    type_col = "'area'"
    name_col = "name"
    id_col = "id"
    if zoom < 6:
        table = "State"
        name_col = "state"
        id_col = "state_id"
    elif zoom < 7:
        table = "SubLocationsLv1"
    elif zoom < 8:
        table = "SubLocationsLv2"
    elif zoom < 9:
        table = "SubLocationsLv3"
    elif zoom < 10:
        table = "SubLocationsLv4"
    elif zoom < 11:
        table = "SubLocationsLv5"
    elif zoom < 12:
        table = "SubLocationsLv6"
    elif zoom < 13:
        table = "SubLocationsLv7"
    elif zoom < 14:
        table = "SubLocationsLv8"
    elif zoom < 15:
        table = "SubLocationsLv9"
    else:
        table = "Routes"
        type_col = "'route'"
        name_col = "route_name"
        id_col = "route_id"

    query = f"""
        SELECT {id_col} as id, {name_col} as name, {lat_col} as latitude, {lng_col} as longitude, {type_col} as type
        FROM {table}
        WHERE {lat_col} BETWEEN %s AND %s
          AND {lng_col} BETWEEN %s AND %s
    """
    try:
        minLat, maxLat = min(swLat, neLat), max(swLat, neLat)
        minLng, maxLng = min(swLng, neLng), max(swLng, neLng)
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(query, (minLat, maxLat, minLng, maxLng))
            results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
