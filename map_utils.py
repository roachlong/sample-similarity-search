import osmnx as ox
import numpy as np
import contextily as cx
import geopandas as gpd
from shapely.geometry import box
from PIL import Image

# --- FETCH REAL-WORLD GRID AND BOUNDS ---
def fetch_grid_and_bounds(place, grid_size):

    # Get the polygon boundary of the place
    gdf = ox.geocode_to_gdf(place)

    # Buffer the polygon outward
    buffered_gdf = gdf.buffer(0.01)  # units are degrees (~1 deg ≈ 111km), so very rough
    
    # Now fetch the graph inside the buffered area
    G = ox.graph_from_polygon(buffered_gdf.unary_union, network_type='drive', truncate_by_edge=False)

    nodes = list(G.nodes(data=True))
    edges = list(G.edges())

    lats = [data['y'] for _, data in nodes]
    lngs = [data['x'] for _, data in nodes]

    north, south = max(lats), min(lats)
    east, west = max(lngs), min(lngs)

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    connections = []

    node_lookup = {node: data for node, data in nodes}
    for u, v in edges:
        if u in node_lookup and v in node_lookup:
            lat1, lon1 = node_lookup[u]['y'], node_lookup[u]['x']
            lat2, lon2 = node_lookup[v]['y'], node_lookup[v]['x']

            x1 = int((lon1 - west) / (east - west) * (grid_size - 1))
            y1 = int((north - lat1) / (north - south) * (grid_size - 1))
            x2 = int((lon2 - west) / (east - west) * (grid_size - 1))
            y2 = int((north - lat2) / (north - south) * (grid_size - 1))

            grid[y1, x1] = 1
            grid[y2, x2] = 1
            connections.append(((y1, x1), (y2, x2)))

    bounds = (north, south, east, west)
    return grid, bounds, connections

# --- FETCH SATELLITE IMAGE ---
def get_satellite_image(bounds, zoom):
    west, south, east, north = bounds[3], bounds[1], bounds[2], bounds[0]
    extent = box(west, south, east, north)
    gdf = gpd.GeoDataFrame({'geometry': [extent]}, crs='EPSG:4326')
    gdf = gdf.to_crs(epsg=3857)

    img, ext = cx.bounds2img(*gdf.total_bounds, zoom=zoom, source=cx.providers.Esri.WorldImagery)
    return Image.fromarray(img).transpose(Image.FLIP_TOP_BOTTOM)