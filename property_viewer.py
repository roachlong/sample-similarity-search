import datetime
from enum import IntEnum
import joblib
from map_utils import fetch_grid_and_bounds, get_satellite_image
import math
import numpy as np
import os
import osmnx as ox
import pickle
from PIL import Image
import psycopg2
import pygame
import networkx as nx
import requests
from tensorflow.keras.models import load_model

PROCESSED_DIR = "data/processed"
MODEL_DIR = "data/models"
SCALER_DIR = "data/scaler"
conn_string_url = os.getenv("DATABASE_URL")

# --- SETTINGS ---
GRID_SIZE = 250
ZOOM_LEVEL = 16
SCREEN_SIZE = 800
SCALE = 1
MIN_SCALE = 0.3
MAX_SCALE = 4.0

class ResultFields(IntEnum):
    MUNICIPALITY = 0
    PLACE = 1
    ADDRESS = 2
    VECTOR = 3

class VectorFields(IntEnum):
    LAT = 0
    LNG = 1
    NO_OF_DWELLINGS = 2
    CORPORATE_OWNED = 3
    ABSENTEE = 4
    UPDATED_AT = 5
    SQ_FT = 6
    BUILDING_CLASS = 7
    YEAR_BUILT = 8
    TAX_RATE = 9
    TAX_RATIO = 10
    RATE_YEAR = 11
    RECORDED_TAXES = 12
    YEAR_1 = 13
    LAND_ASSMT_1 = 14
    BUILDING_ASSMT_1 = 15
    TOTAL_ASSMT_1 = 16
    SALE_PRICE = 17
    SALE_MONTH = 18
    SALE_YEAR = 19
    ADDRESS = 20
    PREDICTION = 21


# --- CONVERSION HELPERS ---
def grid_to_latlon(i, j, grid_size, bounds):
    north, south, east, west = bounds
    lat = north - (i / (grid_size - 1)) * (north - south)
    lon = west + (j / (grid_size - 1)) * (east - west)
    return lat, lon

def grid_to_pixel(i, j, grid_size, image_width, image_height, bounds):
    lat, lon = grid_to_latlon(i, j, grid_size, bounds)
    north, south, east, west = bounds
    x = (lon - west) / (east - west) * (image_width - 1)
    y = (lat - south) / (north - south) * (image_height - 1)
    return x, y

def scale_centered_on_mouse(mouse_pos, old_scale, new_scale, offset_x, offset_y):
    mx, my = mouse_pos
    dx = mx - offset_x
    dy = my - offset_y
    scale_ratio = new_scale / old_scale
    new_offset_x = mx - dx * scale_ratio
    new_offset_y = my - dy * scale_ratio
    return new_offset_x, new_offset_y

def clamp_offset(offset_x, offset_y, scale, image_width, image_height, screen_width, screen_height):
    max_offset_x = 0
    max_offset_y = 0
    min_offset_x = screen_width - image_width * scale
    min_offset_y = screen_height - image_height * scale
    offset_x = max(min(offset_x, max_offset_x), min_offset_x)
    offset_y = max(min(offset_y, max_offset_y), min_offset_y)
    return offset_x, offset_y

def latlng_to_screen(lat, lng, bounds, image_size, scale, offset):
    north, south, east, west = bounds
    w, h = image_size
    x = (lng - west) / (east - west) * w
    y = (lat - south) / (north - south) * h
    screen_x = int(x * scale + offset[0])
    screen_y = int(y * scale + offset[1])
    return screen_x, screen_y

def wrap_text(text, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if font.size(test_line)[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    return lines

def reverse_geocode(lat, lng):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lng,
        "format": "json",
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "YourAppName/1.0"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if 'error' not in data:
            return data.get('display_name', None)  # Nice full address
        else:
            return None
    except Exception as e:
        print(f"Error during reverse geocoding: {e}")
        return None


# --- PREPARE DATA AND LAUNCH A NEW VIEWER ---
def get_building_class_description(conn, building_class_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT description
            FROM property_rating
            WHERE id = (
                SELECT min(round(%s, 20))::int
            );
        """, (building_class_id,))
        result = cur.fetchone()
        return result[0] if result else "Unknown"


def launch_map_viewer(user_input, results):
    if len(results) == 0:
        pygame.quit()
    
    vectors = [row[ResultFields.VECTOR] for row in results]
    average_vector = np.mean(vectors, axis=0)

    # inverse scale the vectorized data
    municipality = results[0][ResultFields.MUNICIPALITY]
    scaler_path = os.path.join(PROCESSED_DIR, f"{municipality}.scaler")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        vectors = [scaler.inverse_transform([vector]).flatten().tolist() for vector in vectors]
        average_vector = scaler.inverse_transform([average_vector]).flatten().tolist()
    else:
        raise FileNotFoundError(f"Could not open the file at {scaler_path}")
    
    # enrich the data
    with psycopg2.connect(conn_string_url) as conn:
        # enhance fields for each vector
        for i, vec in enumerate(vectors):
            building_class_id = vec[VectorFields.BUILDING_CLASS]
            description = get_building_class_description(conn, building_class_id)
            vec[VectorFields.BUILDING_CLASS] = description
            vec.append(results[i][ResultFields.ADDRESS])
        
        # and also for the average vector
        if user_input["lat"] and user_input["lng"]:
            average_vector[VectorFields.LAT] = float(user_input["lat"])
            average_vector[VectorFields.LNG] = float(user_input["lng"])
        average_vector[VectorFields.NO_OF_DWELLINGS] = float(user_input["num_dwellings"])
        average_vector[VectorFields.CORPORATE_OWNED] = float(user_input["corporate_owned"])
        if user_input["square_feet"]:
            average_vector[VectorFields.SQ_FT] = float(user_input["square_feet"])
        if user_input["building_class"]:
            average_vector[VectorFields.BUILDING_CLASS] = float(user_input["building_class"])
        if user_input["yr_built"]:
            average_vector[VectorFields.YEAR_BUILT] = float(user_input["yr_built"])
        if user_input["sale_month"]:
            average_vector[VectorFields.SALE_MONTH] = float(user_input["sale_month"])
        if user_input["sale_year"]:
            average_vector[VectorFields.SALE_YEAR] = float(user_input["sale_year"])
        
        # Integrate AI Model to Predict Home Price using Data from Similarity Search
        model_path = os.path.join(MODEL_DIR, f"{municipality}.keras")
        if os.path.exists(model_path):
            model_scaler_path = os.path.join(SCALER_DIR, municipality + ".features")
            if os.path.exists(model_scaler_path):
                model_scaler = joblib.load(model_scaler_path)
                model_inputs = np.delete(average_vector, VectorFields.SALE_PRICE)
                model_inputs = model_scaler.transform([model_inputs])
                model = load_model(model_path)
                prediction = model.predict(model_inputs)

        building_class_id = average_vector[VectorFields.BUILDING_CLASS]
        description = get_building_class_description(conn, building_class_id)
        average_vector[VectorFields.BUILDING_CLASS] = description
        address = reverse_geocode(average_vector[VectorFields.LAT], average_vector[VectorFields.LNG])
        average_vector.append(address)
        if prediction:
            average_vector.append(prediction[0][0])
    
    place = results[0][ResultFields.PLACE]
    grid_cache_filename = "cache/" + place.replace(",", "_").replace(" ", "_").lower() + "_grid_cache.pkl"
    if os.path.exists(grid_cache_filename):
        with open(grid_cache_filename, "rb") as f:
            grid, bounds, connections = pickle.load(f)
    else:
        grid, bounds, connections = fetch_grid_and_bounds(place, GRID_SIZE)
        with open(grid_cache_filename, "wb") as f:
            pickle.dump((grid, bounds, connections), f)
    
    filename = "cache/" + place.replace(",", "_").replace(" ", "_").lower() + "_satellite.png"
    if os.path.exists(filename):
        pil_img = Image.open(filename)
    else:
        pil_img = get_satellite_image(bounds, ZOOM_LEVEL)
        pil_img.save(filename)
    
    draw_map(place, average_vector, vectors, grid, pil_img, bounds, connections, SCREEN_SIZE, GRID_SIZE, SCALE)


# --- MAIN DRAWING LOOP ---
def draw_map(place, target, comps, grid, sat_image, bounds, connections, screen_size, grid_size, scale=1.0):
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption(f"Comparable Residential Properties in {place}")

    colors = {
        "target": (0, 102, 255),
        "comps": (0, 255, 100),
        "grid": (255, 255, 0),
        "edge": (255, 0, 0),
        "path": (255, 200, 0)
    }

    dragging = False
    last_mouse_pos = (0, 0)
    pan_speed = 10  # for arrow key movement
    zoom_speed = 0.1
    offset_x, offset_y = 0, 0
    sat_surface = pygame.image.fromstring(sat_image.tobytes(), sat_image.size, sat_image.mode)
    active_popup = None
    popup_scroll = 0  # for vertical scroll
    scroll_step = 20
    
    now = datetime.datetime.now()
    current_year = now.year
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    current_month = now.month

    minimap_size = 150
    minimap_surface = pygame.transform.smoothscale(sat_surface, (minimap_size, minimap_size))
    minimap_rect = pygame.Rect(screen_size - minimap_size - 10, 10, minimap_size, minimap_size)
    show_minimap = True

    center_lat = target[VectorFields.LAT]
    center_lng = target[VectorFields.LNG]
    if center_lat is not None and center_lng is not None:
        # Convert center_lat/lng to pixel coordinates
        north, south, east, west = bounds
        x = (center_lng - west) / (east - west) * sat_surface.get_width()
        y = (center_lat - south) / (north - south) * sat_surface.get_height()

        # Center screen on this pixel
        offset_x = screen_size // 2 - int(x * scale)
        offset_y = screen_size // 2 - int(y * scale)

        # Clamp the offsets so we don’t scroll out of bounds
        offset_x, offset_y = clamp_offset(
            offset_x, offset_y, scale,
            sat_surface.get_width(), sat_surface.get_height(),
            screen_size, screen_size
        )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if show_minimap and minimap_rect.collidepoint(event.pos):
                    mx, my = event.pos
                    # Convert click to relative position within the minimap
                    rel_x = (mx - minimap_rect.x) / minimap_size
                    rel_y = (my - minimap_rect.y) / minimap_size

                    # Convert to satellite image pixel coordinates
                    map_x = rel_x * sat_surface.get_width()
                    map_y = rel_y * sat_surface.get_height()

                    # Re-center main view on this point
                    offset_x = screen_size // 2 - int(map_x * scale)
                    offset_y = screen_size // 2 - int(map_y * scale)

                    # Clamp to keep view in bounds
                    offset_x, offset_y = clamp_offset(
                        offset_x, offset_y, scale,
                        sat_surface.get_width(), sat_surface.get_height(),
                        screen_size, screen_size
                    )
                
                elif active_popup and close_btn.collidepoint(event.pos):
                    active_popup = None

                else:
                    # Start dragging if clicked outside minimap
                    dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()

                    # check if pin was clicked
                    mx, my = event.pos
                    pin_x, pin_y = latlng_to_screen(target[VectorFields.LAT],
                                                    target[VectorFields.LNG],
                                                    bounds, (
                                                        sat_surface.get_width(),
                                                        sat_surface.get_height()
                                                    ),
                                                    scale, (offset_x, offset_y))
                    if abs(mx - pin_x) < 8 and abs(my - pin_y) < 8:
                        active_popup = target
                    else:
                        for pin in comps:
                            pin_x, pin_y = latlng_to_screen(pin[VectorFields.LAT],
                                                            pin[VectorFields.LNG],
                                                            bounds, (
                                                                sat_surface.get_width(),
                                                                sat_surface.get_height()
                                                            ),
                                                            scale, (offset_x, offset_y))
                            if abs(mx - pin_x) < 8 and abs(my - pin_y) < 8:
                                active_popup = pin
                                break
            
            elif event.type == pygame.MOUSEWHEEL:
                if active_popup:
                    popup_scroll += event.y * scroll_step

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = pygame.mouse.get_pos()
                dx = mx - last_mouse_pos[0]
                dy = my - last_mouse_pos[1]
                offset_x += dx
                offset_y += dy
                last_mouse_pos = (mx, my)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    show_minimap = not show_minimap

        if dragging:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        keys = pygame.key.get_pressed()

        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:
            new_scale = scale * (1 + zoom_speed)
            offset_x, offset_y = scale_centered_on_mouse(pygame.mouse.get_pos(), scale, new_scale, offset_x, offset_y)
            scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))
        if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]:
            new_scale = scale * (1 - zoom_speed)
            offset_x, offset_y = scale_centered_on_mouse(pygame.mouse.get_pos(), scale, new_scale, offset_x, offset_y)
            scale = max(MIN_SCALE, min(MAX_SCALE, new_scale))
        if keys[pygame.K_LEFT]:
            offset_x += pan_speed
        if keys[pygame.K_RIGHT]:
            offset_x -= pan_speed
        if keys[pygame.K_UP]:
            offset_y += pan_speed
        if keys[pygame.K_DOWN]:
            offset_y -= pan_speed

        offset_x, offset_y = clamp_offset(
            offset_x, offset_y, scale,
            sat_surface.get_width(), sat_surface.get_height(),
            screen_size, screen_size
        )

        # Drawing code
        scaled_img = pygame.transform.smoothscale(sat_surface,
            (int(sat_surface.get_width() * scale), int(sat_surface.get_height() * scale)))
        offset_x, offset_y = clamp_offset(offset_x, offset_y, scale, sat_surface.get_width(), sat_surface.get_height(), screen_size, screen_size)

        screen.blit(scaled_img, (offset_x, offset_y))

        if show_minimap:
            # Mini-map rendering
            screen.blit(minimap_surface, minimap_rect)
            pygame.draw.rect(screen, (255, 255, 255), minimap_rect, 2)  # border

            # Draw viewport rectangle
            map_width = sat_surface.get_width()
            map_height = sat_surface.get_height()

            scaled_map_width = map_width * scale
            scaled_map_height = map_height * scale

            screen_width = screen.get_width()
            screen_height = screen.get_height()

            # Viewport width/height relative to scaled map
            view_w = (screen_width / scaled_map_width) * minimap_size
            view_h = (screen_height / scaled_map_height) * minimap_size

            # Viewport top-left relative to scaled map
            view_x = ((-offset_x) / scaled_map_width) * minimap_size + minimap_rect.x
            view_y = ((-offset_y) / scaled_map_height) * minimap_size + minimap_rect.y

            pygame.draw.rect(screen, (255, 255, 0), pygame.Rect(view_x, view_y, view_w, view_h), 1)

            if target[VectorFields.LAT] is not None and target[VectorFields.LNG] is not None:
                rel_x = (target[VectorFields.LNG] - west) / (east - west)
                rel_y = (target[VectorFields.LAT] - south) / (north - south)

                mini_x = int(rel_x * minimap_size + minimap_rect.x)
                mini_y = int(rel_y * minimap_size + minimap_rect.y)

                # Draw the pin
                pygame.draw.circle(screen, colors['target'], (mini_x, mini_y), 4)
            
            for pin in comps:
                if pin[VectorFields.LAT] is not None and pin[VectorFields.LNG] is not None:
                    rel_x = (pin[VectorFields.LNG] - west) / (east - west)
                    rel_y = (pin[VectorFields.LAT] - south) / (north - south)

                    mini_x = int(rel_x * minimap_size + minimap_rect.x)
                    mini_y = int(rel_y * minimap_size + minimap_rect.y)

                    pygame.draw.circle(screen, colors['comps'], (mini_x, mini_y), 4)

        # drop the target and comp pins
        if target[VectorFields.LAT] is not None and target[VectorFields.LNG] is not None:
            pin_x, pin_y = latlng_to_screen(target[VectorFields.LAT],
                                            target[VectorFields.LNG],
                                            bounds, (
                                                sat_surface.get_width(),
                                                sat_surface.get_height()
                                            ),
                                            scale, (offset_x, offset_y))
            
            pygame.draw.circle(screen, colors['target'], (int(pin_x), int(pin_y)), 8)
        for pin in comps:
            if pin[VectorFields.LAT] is not None and pin[VectorFields.LNG] is not None:
                pin_x, pin_y = latlng_to_screen(pin[VectorFields.LAT],
                                                pin[VectorFields.LNG],
                                                bounds, (
                                                    sat_surface.get_width(),
                                                    sat_surface.get_height()
                                                ),
                                                scale, (offset_x, offset_y))
                pygame.draw.circle(screen, colors['comps'], (int(pin_x), int(pin_y)), 8)

        font = pygame.font.SysFont(None, 24)
        info_text = font.render(f"M = Mini-Map", True, (255, 255, 255))
        screen.blit(info_text, (10, 10))
        
        if active_popup:
            popup_width, popup_height = 300, 250
            popup_x = (screen_size - popup_width) // 2
            popup_y = (screen_size - popup_height) // 2
            padding = 10

            popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
            content_rect = pygame.Rect(popup_x + padding, popup_y + padding, popup_width - 2 * padding, popup_height - 2 * padding)

            pygame.draw.rect(screen, (30, 30, 30), popup_rect)
            pygame.draw.rect(screen, (200, 200, 200), popup_rect, 2)

            font = pygame.font.SysFont(None, 22)
            wrapped_lines = []
            info = active_popup[VectorFields.ADDRESS]
            wrapped_lines.extend(wrap_text(info, font, content_rect.width))
            wrapped_lines.extend(wrap_text(
                "------------------------------------------------",
                font, content_rect.width)
            )

            info = "Sold "
            sale_month = active_popup[VectorFields.SALE_MONTH]
            sale_year = active_popup[VectorFields.SALE_YEAR]
            if sale_year is not None and not math.isnan(sale_year) and current_year <= sale_year:
                if sale_month is not None and not math.isnan(sale_month):
                    if current_month <= sale_month:
                        info = "Could Sell "
            if sale_month is not None and not math.isnan(sale_month):
                if isinstance(sale_month, float):
                    sale_month = math.floor(sale_month)
                info += f"{months[sale_month - 1]}, "
            if sale_year is not None and not math.isnan(sale_year):
                if isinstance(sale_year, float):
                    sale_year = math.floor(sale_year)
                info += f"{sale_year} for "
            else:
                info += "???? for "
            sale_price = active_popup[VectorFields.SALE_PRICE]
            if sale_price is not None and not math.isnan(sale_price):
                info += f"${int(float(sale_price)):,}"
            else:
                info += "$###"
            wrapped_lines.extend(wrap_text(info, font, content_rect.width))

            info = ""
            num_dwellings = active_popup[VectorFields.NO_OF_DWELLINGS]
            if num_dwellings is not None and not math.isnan(num_dwellings):
                if isinstance(num_dwellings, float):
                    num_dwellings = math.floor(num_dwellings)
                info += f"{num_dwellings} Dwelling(s) "
            square_feet = active_popup[VectorFields.SQ_FT]
            if square_feet is not None and not math.isnan(square_feet):
                info += f"with {int(float(square_feet)):,} sq ft"
            wrapped_lines.extend(wrap_text(info, font, content_rect.width))

            if active_popup[VectorFields.CORPORATE_OWNED]:
                info = "Corporate Owned"
            else:
                info = "Privately Owned"
            wrapped_lines.extend(wrap_text(info, font, content_rect.width))

            info = active_popup[VectorFields.BUILDING_CLASS]
            yr_built = active_popup[VectorFields.YEAR_BUILT]
            if yr_built is not None and not math.isnan(yr_built):
                if isinstance(yr_built, float):
                    yr_built = math.floor(yr_built)
                info += f" built {yr_built}"
            wrapped_lines.extend(wrap_text(info, font, content_rect.width))

            info = "Taxes"
            taxes_paid = active_popup[VectorFields.RECORDED_TAXES]
            if taxes_paid is not None and not math.isnan(taxes_paid):
                info += f" paid ${int(float(taxes_paid)):,}"
            tax_rate = active_popup[VectorFields.TAX_RATE]
            if tax_rate is not None and not math.isnan(tax_rate):
                info += f" at {float(tax_rate):.2f}%"
            wrapped_lines.extend(wrap_text(info, font, content_rect.width))

            if len(active_popup) > 21:
                prediction = active_popup[VectorFields.PREDICTION]
                if prediction is not None and not math.isnan(prediction):
                    info = f"Predicted Value: ${int(float(prediction)):,}"
                    wrapped_lines.extend(wrap_text(info, font, content_rect.width))

            # Clamp scrolling
            total_height = len(wrapped_lines) * 24
            max_scroll = max(0, total_height - content_rect.height)
            popup_scroll = max(0, min(popup_scroll, max_scroll))

            # Clip to content rect
            clip_surface = pygame.Surface((content_rect.width, content_rect.height), pygame.SRCALPHA).convert_alpha()
            clip_surface.fill((30, 30, 30))

            for i, line in enumerate(wrapped_lines):
                line_y = i * 24 - popup_scroll
                if 0 <= line_y < content_rect.height:
                    txt = font.render(line, True, (255, 255, 255))
                    clip_surface.blit(txt, (0, line_y))

            screen.blit(clip_surface, content_rect.topleft)

            # Draw close button
            close_btn = pygame.Rect(popup_x + popup_width - 30, popup_y + 10, 20, 20)
            pygame.draw.rect(screen, (255, 0, 0), close_btn)
            pygame.draw.line(screen, (255, 255, 255), close_btn.topleft, close_btn.bottomright, 2)
            pygame.draw.line(screen, (255, 255, 255), close_btn.topright, close_btn.bottomleft, 2)

        pygame.display.flip()

    pygame.quit()

