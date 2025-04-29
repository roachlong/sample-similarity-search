from enum import IntEnum
import joblib
import os
import psycopg2

PROCESSED_DIR = "data/processed"
conn_string_url = os.getenv("DATABASE_URL")

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

def perform_search(user_input):
    municipality = user_input["municipality"]
    embedding = vectorize_input(municipality, user_input.copy())  # Must return a list or numpy array
    placeholders = ', '.join(['%s'] * len(embedding))
    vector_sql = f'array[{placeholders}]::vector'

    query = f"""
        SELECT v.municipality,
               concat(
                   r.property_data->'countyData'->>'P_City', ', ',
                   r.property_data->'countyData'->>'P_State', ' ',
                   r.property_data->'countyData'->>'P_Zip'
               ) as place,
               r.property_data->'address' as address,
               v.vector_data::float8[] as vector
        FROM property_vectors v
        JOIN raw_property_data r ON r.id = v.id
        WHERE v.municipality = %s
        ORDER BY v.vector_data <#> {vector_sql}
        LIMIT 10;
    """

    with psycopg2.connect(conn_string_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [municipality] + embedding)
            results = cur.fetchall()
            return results


def vectorize_input(municipality, user_input):
    query = f"""
        SELECT avg(lat) AS lat,
            avg(lng) AS lng,
            avg(CASE
                    WHEN no_of_dwellings IS NOT NULL AND no_of_dwellings != 'NaN' THEN no_of_dwellings
                    ELSE NULL
                END
            ) AS no_of_dwellings,
            round(avg(CASE
                WHEN corporate_owned IS NULL
                THEN NULL
                ELSE CASE
                    WHEN corporate_owned THEN 1
                    ELSE 0
                END
            END)) AS corporate_owned,
            avg(absentee) AS absentee,
            avg(EXTRACT(EPOCH FROM updated_at)) AS updated_at,
            avg(CASE
                    WHEN sq_ft IS NOT NULL AND sq_ft != 'NaN' THEN sq_ft
                    ELSE NULL
                END
            ) AS sq_ft,
            avg(CASE
                WHEN building_class !~ '^(\+|-)?[[:digit:]]+(\.)?[[:digit:]]+$'
                THEN NULL
                ELSE building_class::FLOAT8
            END) AS building_class,
            avg(CASE
                    WHEN yr_built IS NOT NULL AND yr_built != 'NaN' THEN yr_built
                    ELSE NULL
                END
            ) AS yr_built,
            avg(taxrate) AS taxrate,
            avg(taxratio) AS taxratio,
            avg(rateyear) AS rateyear,
            avg(CASE
                    WHEN recorded_taxes IS NOT NULL AND recorded_taxes != 'NaN' THEN recorded_taxes
                    ELSE NULL
                END
            ) AS recorded_taxes,
            avg(year_1) AS year_1,
            avg(land_assmnt_1) AS land_assmnt_1,
            avg(building_assmnt_1) AS building_assmnt_1,
            avg(total_assmnt_1) AS total_assmnt_1,
            avg(sale_price) AS sale_price
        FROM property_vectors
        WHERE municipality = '{municipality}'
    """

    if user_input["num_dwellings"]:
        query += f"""
            AND no_of_dwellings >= {user_input['num_dwellings'] - 1}
            AND no_of_dwellings <= {user_input['num_dwellings'] + 1}
        """
    if user_input["corporate_owned"]:
        query += f"""
            AND corporate_owned = {user_input['corporate_owned']}
        """
    if user_input["square_feet"]:
        query += f"""
            AND sq_ft >= {user_input['square_feet'] * 0.9}
            AND sq_ft <= {user_input['square_feet'] * 1.1}
        """
    if user_input["building_class"]:
        query += f"""
            AND (
                building_class = '{user_input['building_class'] - 1}'
                OR building_class = '{user_input['building_class']}'
                OR building_class = '{user_input['building_class'] + 1}'
            )
        """
    if user_input["yr_built"]:
        query += f"""
            AND yr_built >= {user_input['yr_built'] - 3}
            AND yr_built <= {user_input['yr_built'] + 3}
        """
    if user_input["recorded_taxes"]:
        query += f"""
            AND recorded_taxes >= {user_input['recorded_taxes'] * 0.9}
            AND recorded_taxes <= {user_input['recorded_taxes'] * 1.1}
        """

    with psycopg2.connect(conn_string_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query,)
            average_input = cur.fetchone()

    if average_input[VectorFields.LAT]:
        if not user_input["lat"]:
            user_input["lat"] = average_input[VectorFields.LAT]
        if not user_input["lng"]:
            user_input["lng"] = average_input[VectorFields.LNG]
        if not user_input["num_dwellings"]:
            user_input["num_dwellings"] = average_input[VectorFields.NO_OF_DWELLINGS]
        user_input["absentee"] = average_input[VectorFields.ABSENTEE]
        user_input["updated_at"] = average_input[VectorFields.UPDATED_AT]
        if not user_input["square_feet"]:
            user_input["square_feet"] = average_input[VectorFields.SQ_FT]
        if not user_input["building_class"]:
            user_input["building_class"] = average_input[VectorFields.BUILDING_CLASS]
        if not user_input["yr_built"]:
            user_input["yr_built"] = average_input[VectorFields.YEAR_BUILT]
        user_input["tax_rate"] = average_input[VectorFields.TAX_RATE]
        user_input["tax_ratio"] = average_input[VectorFields.TAX_RATIO]
        user_input["rate_year"] = average_input[VectorFields.RATE_YEAR]
        if not user_input["recorded_taxes"]:
            user_input["recorded_taxes"] = average_input[VectorFields.RECORDED_TAXES]
        user_input["year_1"] = average_input[VectorFields.YEAR_1]
        user_input["land_assmnt_1"] = average_input[VectorFields.LAND_ASSMT_1]
        user_input["building_assmnt_1"] = average_input[VectorFields.BUILDING_ASSMT_1]
        user_input["total_assmnt_1"] = average_input[VectorFields.TOTAL_ASSMT_1]
        user_input["sale_price"] = average_input[VectorFields.SALE_PRICE]
    else:
        if not user_input["lat"]:
            user_input["lat"] = 0
        if not user_input["lng"]:
            user_input["lng"] = 0
        if not user_input["num_dwellings"]:
            user_input["num_dwellings"] = 0
        user_input["absentee"] = 0
        user_input["updated_at"] = 0
        if not user_input["square_feet"]:
            user_input["square_feet"] = 0
        if not user_input["building_class"]:
            user_input["building_class"] = 0
        if not user_input["yr_built"]:
            user_input["yr_built"] = 0
        user_input["tax_rate"] = 0
        user_input["tax_ratio"] = 0
        user_input["rate_year"] = 0
        if not user_input["recorded_taxes"]:
            user_input["recorded_taxes"] = 0
        user_input["year_1"] = 0
        user_input["land_assmnt_1"] = 0
        user_input["building_assmnt_1"] = 0
        user_input["total_assmnt_1"] = 0
        user_input["sale_price"] = 0
    
    vector = [
        user_input["lat"],
        user_input["lng"],
        user_input["num_dwellings"],
        user_input["corporate_owned"],
        user_input["absentee"],
        user_input["updated_at"],
        user_input["square_feet"],
        user_input["building_class"],
        user_input["yr_built"],
        user_input["tax_rate"],
        user_input["tax_ratio"],
        user_input["rate_year"],
        user_input["recorded_taxes"],
        user_input["year_1"],
        user_input["land_assmnt_1"],
        user_input["building_assmnt_1"],
        user_input["total_assmnt_1"],
        user_input["sale_price"],
        user_input["sale_month"],
        user_input["sale_year"]
    ]

    # convert vector inputs to float
    vector = [float(v) for v in list(vector)]

    # scale the vectorized data
    scaler_path = os.path.join(PROCESSED_DIR, f"{municipality}.scaler")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        vector = scaler.transform([vector])
    else:
        raise FileNotFoundError(f"Could not open the file at {scaler_path}")
        
    return vector.flatten().tolist()
