# app/routers/map_data.py
# ------------------------------------------------------------------------
# This module provides an endpoint to fetch map data from the database.
# It retrieves sub-locations, routes, and state data with coordinates.
# The data is structured to include latitude and longitude for each feature.
# ------------------------------------------------------------------------

from fastapi import APIRouter, HTTPException
import logging

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/map-data")
def get_map_data():
    try:
        # Import here to catch import errors
        try:
            from app.storage.database.route_db_connect import get_connection
            logger.info("Successfully imported get_connection")
        except ImportError as e:
            logger.error(f"Failed to import get_connection: {e}")
            raise HTTPException(status_code=500, detail=f"Database module import failed: {e}")

        # Test database connection
        try:
            conn = get_connection()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

        try:
            cursor = conn.cursor()
            results = []

            logger.info("Querying data with POINT geometry coordinates...")

            # Query sub-location levels to fetch coordinates
            # Loop through levels 1 to 10 to get all sub-locations
            for level in range(1, 11):
                try:
                    subloc_query = f"""
                        SELECT 
                            location_id AS id, 
                            location_name AS name, 
                            'area' AS type, 
                            parent_id, 
                            NULL AS rating,
                            ST_Y(coordinates) AS latitude,
                            ST_X(coordinates) AS longitude
                        FROM SubLocationsLv{level}
                        WHERE coordinates IS NOT NULL
                        LIMIT 100
                    """
                    
                    cursor.execute(subloc_query)
                    subloc_results = cursor.fetchall()
                    
                    # Check if results are dictionaries or tuples
                    if subloc_results and isinstance(subloc_results[0], dict):
                        # Convert dict results to a list of dicts
                        results.extend(subloc_results)
                        logger.info(f"Found {len(subloc_results)} areas in SubLocationsLv{level} with coordinates")
                        # Log first result to verify coordinates
                        if subloc_results:
                            first = subloc_results[0]
                            logger.info(f"Sample result: {first['name']} - lat: {first['latitude']}, lng: {first['longitude']}")
                    # If results are tuples, convert to dicts
                    else:
                        for row in subloc_results:
                            results.append({
                                "id": row[0],
                                "name": row[1],
                                "type": row[2],
                                "parent_id": row[3],
                                "rating": row[4],
                                "latitude": row[5],
                                "longitude": row[6]
                            })
                        
                        # Log the number of results found
                        if subloc_results:
                            logger.info(f"Found {len(subloc_results)} areas in SubLocationsLv{level}")
                    
                except Exception as e:
                    logger.debug(f"No data in SubLocationsLv{level}: {e}")

            # Create a mapping of area IDs to coordinates for route inheritance
            area_coordinates = {}
            for item in results:
                if item['type'] == 'area' and item['latitude'] is not None and item['longitude'] is not None:
                    area_coordinates[item['id']] = {
                        'latitude': item['latitude'],
                        'longitude': item['longitude']
                    }

            logger.info(f"Built coordinate map for {len(area_coordinates)} areas")

            # Query Routes and inherit coordinates from parent areas
            try:
                routes_query = """
                    SELECT route_id AS id, route_name AS name, 'route' AS type, parent_id, rating,
                           NULL AS latitude, NULL AS longitude
                    FROM Routes
                    LIMIT 500
                """
                cursor.execute(routes_query)
                route_results = cursor.fetchall()
                
                routes_with_coords = 0
                if route_results and isinstance(route_results[0], dict):
                    for route in route_results:
                        parent_id = route.get('parent_id')
                        if parent_id and parent_id in area_coordinates:
                            route['latitude'] = area_coordinates[parent_id]['latitude']
                            route['longitude'] = area_coordinates[parent_id]['longitude']
                            routes_with_coords += 1
                        results.append(route)
                else:
                    for row in route_results:
                        route = {
                            "id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "parent_id": row[3],
                            "rating": row[4],
                            "latitude": row[5],
                            "longitude": row[6]
                        }
                        
                        # Inherit coordinates from parent area
                        parent_id = route['parent_id']
                        if parent_id and parent_id in area_coordinates:
                            route['latitude'] = area_coordinates[parent_id]['latitude']
                            route['longitude'] = area_coordinates[parent_id]['longitude']
                            routes_with_coords += 1
                        
                        results.append(route)
                
                logger.info(f"Found {len(route_results)} routes, {routes_with_coords} with inherited coordinates")
                
            except Exception as e:
                logger.warning(f"Failed to query routes: {e}")

            # Query State level if it exists
            try:
                state_query = """
                    SELECT 
                        state_id AS id, 
                        state AS name, 
                        'state' AS type, 
                        NULL AS parent_id, 
                        NULL AS rating,
                        ST_Y(coordinates) AS latitude,
                        ST_X(coordinates) AS longitude
                    FROM State
                    WHERE coordinates IS NOT NULL
                """
                cursor.execute(state_query)
                state_results = cursor.fetchall()
                
                if state_results and isinstance(state_results[0], dict):
                    results.extend(state_results)
                    logger.info(f"Found {len(state_results)} states with coordinates")
                else:
                    for row in state_results:
                        results.append({
                            "id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "parent_id": row[3],
                            "rating": row[4],
                            "latitude": row[5],
                            "longitude": row[6]
                        })
                    
                    if state_results:
                        logger.info(f"Found {len(state_results)} states")
                        
            except Exception as e:
                logger.info(f"No State table or no coordinates: {e}")

            logger.info(f"Returning {len(results)} total features")
            
            return {
                "features": results,
                "total_count": len(results)
            }

        except Exception as e:
            logger.exception("Database query failed")
            raise HTTPException(status_code=500, detail=f"Database query failed: {e}")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in get_map_data")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    finally:
        try:
            if 'conn' in locals():
                conn.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
