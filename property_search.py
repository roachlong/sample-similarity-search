import datetime
import os
import PySimpleGUI as sg
from property_service import perform_search
from property_viewer import launch_map_viewer
import psycopg2
import requests
import time

conn_string_url = os.getenv("DATABASE_URL")

def validate_address(address):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "YourAppName/1.0"
    }
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        if data:
            lat = data[0]['lat']
            lon = data[0]['lon']
            return True, lat, lon
        else:
            return False, None, None
    except Exception as e:
        print(f"Error validating address: {e}")
        return False, None, None


def main():

    address_cache = {}  # {address: (is_valid, lat, lng)}
    last_address_edit_time = None
    address_typing_timeout = 0.5  # seconds
    now = datetime.datetime.now()
    current_year = now.year
    years = list(range(current_year + 1, 1900, -1))  # From this year down to 1900
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    current_month = now.month

    layout = [
        [sg.Text("County Search:"),
         sg.Input(key="county_search", enable_events=True)],
        [sg.Listbox(values=[], size=(40, 3), key="county_suggestions", enable_events=True, disabled=True)],
        [sg.Text("District Search:"),
         sg.Input(key="district_search", enable_events=True, disabled=True)],
        [sg.Listbox(values=[], size=(40, 3), key="district_suggestions", enable_events=True, disabled=True)],
        [sg.Input(key="municipality", visible=False)],
        [sg.Text("Address:"),
         sg.Input(key="address_input", enable_events=True, disabled=True)],
        [sg.Text("", size=(40, 1), key="address_feedback")],
        [sg.Input(key="address_lat", visible=False)],
        [sg.Input(key="address_lng", visible=False)],
        [sg.Text("Number of Dwellings:"),
         sg.Combo(values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], key="num_dwellings", default_value=1, readonly=True)],
        [sg.Checkbox("Corporate Owned", key="corporate_owned")],
        [sg.Text("Square Feet:"),
         sg.Input(key="sq_ft_input", enable_events=True)],
        [sg.Text("Building Class:"),
         sg.Combo(values=[], size=(40, 1), key="building_class", readonly=True)],
        [sg.Text("Year Built:"),
         sg.Combo(values=years, key="yr_built", readonly=True)],
        [sg.Text("Estimated Taxes:"),
         sg.Input(key="tax_input", enable_events=True)],
        [sg.Text("Month of Sale:"),
         sg.Combo(values=months, key="sale_month", default_value=months[current_month - 1], readonly=True)],
        [sg.Text("Year of Sale:"),
         sg.Combo(values=years, key="sale_year", default_value=current_year, readonly=True)],
        [sg.Button("Submit", key="submit_button", disabled=True),
         sg.Button("Clear Form", key="clear_button"),
         sg.Button("Exit")]
    ]

    window = sg.Window("Property Search", layout)

    # 1st read — initialize the window and widgets
    event, values = window.read(timeout=0)

    with psycopg2.connect(conn_string_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, description
                FROM property_rating
                ORDER BY id
            """)
            building_class_map = {description: id for id, description in cur.fetchall()}
        
        window["building_class"].update(values=list(building_class_map.keys()))

        while True:
            event, values = window.read(timeout=50)

            if event == sg.WIN_CLOSED or event == "Exit":
                break

            if event == "clear_button":
                # Reset optional fields manually
                window["address_input"].update("")
                window["address_feedback"].update("")
                window["address_lat"].update("")
                window["address_lng"].update("")
                window["num_dwellings"].update("1")
                window["corporate_owned"].update(False)
                window["sq_ft_input"].update("")
                window["building_class"].update("")
                window["yr_built"].update("")
                window["tax_input"].update("")
                window["sale_month"].update(f"{months[current_month - 1]}")
                window["sale_year"].update(f"{current_year}")

            elif event == "submit_button":
                search_input = {
                    "municipality": values["municipality"],
                    "lat": values["address_lat"],
                    "lng": values["address_lng"],
                    "num_dwellings": values["num_dwellings"],
                    "corporate_owned": values["corporate_owned"],
                    "square_feet": int(values["sq_ft_input"]) if values["sq_ft_input"] else 0,
                    "building_class": building_class_map[values["building_class"]] if values["building_class"] else 0,
                    "yr_built": int(values["yr_built"]) if values["yr_built"] else 0,
                    "recorded_taxes": int(values["tax_input"]) if values["tax_input"] else 0,
                    "sale_month": months.index(values["sale_month"]) + 1 if values["sale_month"] else 0,
                    "sale_year": int(values["sale_year"]) if values["sale_year"] else 0
                }
                results = perform_search(search_input)
                if results:
                    launch_map_viewer(search_input, results)

            elif event == "county_search":
                search_term = values["county_search"].lower()
                window["county_suggestions"].update(disabled=False)
                window["district_search"].update(disabled=True)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT county_name
                        FROM property_location
                        WHERE lower(county_name) LIKE %s
                        ORDER BY county_name
                        LIMIT 10
                    """, (f"%{search_term}%",))
                    results = [row[0] for row in cur.fetchall()]

                window["county_suggestions"].update(results)

            elif event == "county_suggestions":
                selected = values["county_suggestions"][0]
                window["county_search"].update(selected)
                window["district_search"].update(disabled=False)

            elif event == "district_search":
                county_name = values["county_search"].lower()
                search_term = values["district_search"].lower()
                window["district_suggestions"].update(disabled=False)
                window["address_input"].update(disabled=True)
                window["submit_button"].update(disabled=True)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT district_name
                        FROM property_location
                        WHERE lower(county_name) = %s
                          AND lower(district_name) like %s
                        ORDER BY district_name
                        LIMIT 10
                    """, (county_name, f"%{search_term}%",))
                    results = [row[0] for row in cur.fetchall()]

                window["district_suggestions"].update(results)

            elif event == "district_suggestions":
                selected = values["district_suggestions"][0]
                window["district_search"].update(selected)

                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT municipality
                        FROM property_location
                        WHERE district_name = %s
                        LIMIT 1
                    """, (selected,))
                    result = cur.fetchone()
                    if result:
                        municipality = result[0]
                        window["municipality"].update(municipality)
                        window["address_input"].update(disabled=False)
                        window["submit_button"].update(disabled=False)
            
            elif event == "address_input":
                last_address_edit_time = time.time()
                window["address_feedback"].update("Typing...", text_color="gray")
                window["address_lat"].update("")
                window["address_lng"].update("")

            elif event == "sq_ft_input":
                value = values["sq_ft_input"]
                # Only allow digits
                if not value.isdigit():
                    # Remove non-digit characters
                    cleaned_value = ''.join(filter(str.isdigit, value))
                    window["sq_ft_input"].update(cleaned_value)

            elif event == "tax_input":
                value = values["tax_input"]
                # Only allow digits
                if not value.isdigit():
                    # Remove non-digit characters
                    cleaned_value = ''.join(filter(str.isdigit, value))
                    window["tax_input"].update(cleaned_value)

            
            # Handle delayed validation after typing stops
            current_time = time.time()
            if last_address_edit_time and (current_time - last_address_edit_time) >= address_typing_timeout:
                address = values['address_input']
                last_address_edit_time = None  # reset

                if len(address) < 5:
                    window["address_feedback"].update("Address too short...", text_color="orange")
                else:
                    town = values["district_search"].split()[0] if values["district_search"] else ""
                    address = f"{address} {town}, NJ"
                    if address in address_cache:
                        is_valid, lat, lng = address_cache[address]
                    else:
                        is_valid, lat, lng = validate_address(address)
                        address_cache[address] = (is_valid, lat, lng)

                    if is_valid:
                        window["address_feedback"].update("✅ Address found!", text_color="green")
                        window["address_lat"].update(lat)
                        window["address_lng"].update(lng)
                    else:
                        window["address_feedback"].update("❌ Address not found.", text_color="red")
                        window["address_lat"].update("")
                        window["address_lng"].update("")

    window.close()

if __name__ == '__main__':
    main()
