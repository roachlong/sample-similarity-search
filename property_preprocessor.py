from filelock import FileLock
import joblib
import logging
import numpy as np
import os
import pandas as pd
from pandas import json_normalize
import pickle
import psycopg2
from psycopg2 import OperationalError, DatabaseError
from psycopg2.extras import execute_values
from sklearn.preprocessing import MinMaxScaler
import time

PROCESSED_DIR = "data/processed"
conn_string_url = os.getenv("DATABASE_URL")
extra_columns = [
    "lat", "lng", "No_Of_Dwellings", "Corporate_Owned", "Absentee",
    "updated_at", "Sq_Ft", "Building_Class", "Yr_Built", "TaxRate",
    "TaxRatio", "RateYear", "Recorded_Taxes", "Year_1", "Land_Assmnt_1",
    "Building_Assmnt_1", "Total_Assmnt_1", "Sale_Price"
]

def upsert_property_vectors(municipality, id_vector_map, original_df, max_retries=5, base_delay=1.0):
    """
    Inserts or updates property_vectors using multi-value insert with retries on serialization failures.

    :param id_vector_map: dict[UUID -> List[float]] with vectors of length 20
    :param max_retries: Maximum number of retry attempts
    :param base_delay: Base delay between retries (exponential backoff)
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            with psycopg2.connect(conn_string_url) as conn:
                with conn.cursor() as cur:
                    records = []
                    for uuid, vector in id_vector_map.items():
                        vec = [float(v) for v in list(vector)]
                        if len(vec) != 20:
                            logging.warning(f"Skipping {uuid} — vector has invalid length: {len(vec)}")
                            continue

                        # Fetch corresponding row from original_df
                        try:
                            row = original_df.loc[original_df['id'] == uuid].iloc[0]
                            row = reduce_columns(row.to_frame().T).iloc[0]
                        except IndexError:
                            logging.warning(f"Skipping {uuid} — not found in original_df.")
                            continue

                        # Extract extra column values
                        extras = [convert_value(row.get(col, None)) for col in extra_columns]

                        # Append full record (uuid, municipality, vector, plus extras)
                        records.append((uuid, municipality, vec, *extras))

                    insert_query = f"""
                        INSERT INTO property_vectors (
                            id, municipality, vector_data, {', '.join(extra_columns)}
                        )
                        VALUES %s
                        ON CONFLICT (id) DO UPDATE SET
                            municipality = EXCLUDED.municipality,
                            vector_data = EXCLUDED.vector_data,
                            {', '.join([f"{col} = EXCLUDED.{col}" for col in extra_columns])}
                    """
                    
                    template = "(" + ",".join(["%s"] * (3 + len(extra_columns))) + ")"
                    execute_values(
                        cur,
                        insert_query,
                        records,
                        template=template
                    )

                    conn.commit()
                    logging.info(f"Upserted {len(records)} property vectors for {municipality}.")
                    return  # Success — exit the function

        except (OperationalError, DatabaseError) as e:
            if "retry transaction" in str(e).lower() or "serialization failure" in str(e).lower():
                logging.warning(f"Retryable error on attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(base_delay * (2 ** attempt))  # Exponential backoff
                attempt += 1
            else:
                logging.exception("Non-retryable database error during upsert.")
                raise  # Re-raise non-retryable exceptions

    logging.error(f"Failed to upsert property vectors after {max_retries} attempts.")
    raise RuntimeError("Max retries exceeded for property vector upsert.")


def process_batch(municipality, messages):
    try:
        # vectorize the data
        original_df = json_normalize(messages)
        vector_df = vectorize_data(original_df)
        original_df = original_df.loc[vector_df.index].copy()

        # scale the vectorized data
        scaler_path = os.path.join(PROCESSED_DIR, f"{municipality}.scaler")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            vector = scaler.transform(vector_df.values)
        else:
            scaler = MinMaxScaler()
            vector = scaler.fit_transform(vector_df.values)
            joblib.dump(scaler, scaler_path)
        
        original_df['vector'] = vector.tolist()

        # save the vectorized input
        id_vector_map = dict(zip(original_df['id'], original_df['vector']))
        upsert_property_vectors(municipality, id_vector_map, original_df)

        # Pickle path and lock file
        pickle_path = os.path.join(PROCESSED_DIR, f"{municipality}.pkl")
        lock_path = f"{pickle_path}.lock"
        lock = FileLock(lock_path)
        
        with lock:
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    existing_df = pickle.load(f)  # Expect this to be a DataFrame
            else:
                existing_df = pd.DataFrame()

            # Combine existing and new
            combined_df = pd.concat([existing_df, vector_df], ignore_index=True)

            # Write back to file
            with open(pickle_path, 'wb') as f:
                pickle.dump(combined_df, f)

    except Exception as e:
        logging.exception(f"Failed processing batch for {municipality}")


def convert_value(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()  # for timestamp fields
    return v


def reduce_columns(df):
    columns_to_keep = [
        'property_data.location.lat',
        'property_data.location.lng',
        'property_data.countyData.No_Of_Dwellings',
        'property_data.countyData.Corporate_Owned',
        'property_data.countyData.Absentee',
        'property_data.countyData.NU_Code',
        'property_data.countyData.updated_at',
        'property_data.countyData.TotalUnits',
        'property_data.countyData.Sq_Ft',
        'property_data.countyData.Property_Class',
        'property_data.countyData.Building_Class',
        'property_data.countyData.Yr_Built',
        'property_data.countyData.Sale_Date',
        'property_data.countyData.TaxRate',
        'property_data.countyData.TaxRatio',
        'property_data.countyData.RateYear',
        'property_data.countyData.Recorded_Taxes',
        'property_data.countyData.Calculated_Taxes',
        'property_data.countyData.Calculated_Taxes_Year',
        'property_data.countyData.Year_1',
        'property_data.countyData.Land_Assmnt_1',
        'property_data.countyData.Building_Assmnt_1',
        'property_data.countyData.Total_Assmnt_1',
        'property_data.countyData.Sale_Price'
    ]

    return df[columns_to_keep].rename(columns={
        'property_data.location.lat': 'lat',
        'property_data.location.lng': 'lng',
        'property_data.countyData.No_Of_Dwellings': 'No_Of_Dwellings',
        'property_data.countyData.Corporate_Owned': 'Corporate_Owned',
        'property_data.countyData.Absentee': 'Absentee',
        'property_data.countyData.NU_Code': 'NU_Code',
        'property_data.countyData.updated_at': 'updated_at',
        'property_data.countyData.TotalUnits': 'TotalUnits',
        'property_data.countyData.Sq_Ft': 'Sq_Ft',
        'property_data.countyData.Property_Class': 'Property_Class',
        'property_data.countyData.Building_Class': 'Building_Class',
        'property_data.countyData.Yr_Built': 'Yr_Built',
        'property_data.countyData.Sale_Date': 'Sale_Date',
        'property_data.countyData.TaxRate': 'TaxRate',
        'property_data.countyData.TaxRatio': 'TaxRatio',
        'property_data.countyData.RateYear': 'RateYear',
        'property_data.countyData.Recorded_Taxes': 'Recorded_Taxes',
        'property_data.countyData.Calculated_Taxes': 'Calculated_Taxes',
        'property_data.countyData.Calculated_Taxes_Year': 'Calculated_Taxes_Year',
        'property_data.countyData.Year_1': 'Year_1',
        'property_data.countyData.Land_Assmnt_1': 'Land_Assmnt_1',
        'property_data.countyData.Building_Assmnt_1': 'Building_Assmnt_1',
        'property_data.countyData.Total_Assmnt_1': 'Total_Assmnt_1',
        'property_data.countyData.Sale_Price': 'Sale_Price'
    })


def vectorize_data(df):
    features = reduce_columns(df)
    features = features.dropna(subset=[
        'Land_Assmnt_1', 'Building_Assmnt_1', 'Total_Assmnt_1', 'lat', 'lng'
    ])

    # only keep records that are not assigned a non-usable code
    features['NU_Code'] = features['NU_Code'].apply(
        lambda x: int(x) if pd.notna(x) and x.isdigit() else -1 if pd.notna(x) else 99
    )
    features = features[features['NU_Code'] == 99]
    features = features.drop('NU_Code', axis=1)
    
    # convert date to numerical value
    features['updated_at'] = pd.to_datetime(
        features['updated_at']
    ).values.astype('datetime64[ms]').astype(int)
    features['updated_at'] = features['updated_at'].fillna(0)
    
    # should only have residential properties from our original dataset
    features['Property_Class'] = features['Property_Class'].map({
        '1': 1, '2': 2, '3A': 3, '3B': 4, '4A': 5, '4B': 6, '4C': 7, '5A': 8, '5B': 9, '6A': 10,
        '6B': 11, '6C': 12, '15A': 13, '15B': 14, '15C': 15, '15D': 16, '15E': 17, '15F': 18
    })
    features = features[features['Property_Class'] == 2]
    features = features.drop(['Property_Class', 'TotalUnits'], axis=1)
    
    # keep the month and year of the property sale date
    features['Sale_Date'] = pd.to_datetime(features['Sale_Date'])
    features['Sale_Month'] = features['Sale_Date'].dt.month.astype(pd.Int64Dtype())
    features['Sale_Year'] = features['Sale_Date'].dt.year.astype(pd.Int64Dtype())
    features = features.drop('Sale_Date', axis=1)
    features = features.dropna(subset=['Sale_Month', 'Sale_Year'])

    # convert values to numerical type
    features['TaxRate'] = features['TaxRate'].astype(float)
    features['TaxRatio'] = features['TaxRatio'].astype(float)
    features['RateYear'] = features['RateYear'].astype(int)

    # consolidate building class to common values
    features['Building_Class'] = features['Building_Class'].apply(
        lambda x: convert_building_class(x)).astype(pd.Int64Dtype()
    )
    update_mapping = pd.DataFrame({
        'key':[
            26, 27, 28, 29, 30,
            33, 34, 35, 36, 37, 38, 39,
            43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53, 54, 55
        ], 'value':[
            12, 13, 14, 16, 18,
            13, 14, 15, 16, 17, 18, 19,
            13, 14, 15, 16, 17, 18, 19,
            12, 13, 15, 18, 19, 20
        ]
    })
    features['Building_Class'] = features['Building_Class'].replace(
        dict(zip(update_mapping['key'], update_mapping['value']))
    )
    avg_bld_class = features[features['Building_Class'].notna()].groupby(
        'Total_Assmnt_1'
    ).mean()['Building_Class']
    features['Building_Class'] = features.apply(
        lambda x: fill_building_class(x['Total_Assmnt_1'],
                                      x['Building_Class'],
                                      avg_bld_class), axis=1
    )
    features['Building_Class'] = features['Building_Class'].astype(
        pd.Float64Dtype()
    ).round().astype(pd.Int64Dtype())
    
    # impute missing values for square feet
    avg_feet = features[features['Sq_Ft'].notna()].groupby('Building_Assmnt_1').mean()['Sq_Ft']
    features['Sq_Ft'] = features.apply(
        lambda x: fill_square_feet(x['Building_Assmnt_1'], x['Sq_Ft'], avg_feet), axis=1
    )

    # impute misisng values for number of dwellings
    avg_dwellings = features[features['No_Of_Dwellings'].notna()].groupby(
        ['Building_Class', 'Sq_Ft']
    ).agg(
        {'No_Of_Dwellings': 'mean'}
    ).reset_index()
    features['No_Of_Dwellings'] = features.apply(
        lambda x: fill_no_of_dwellings(x['Building_Class'],
                                       x['Sq_Ft'], x['No_Of_Dwellings'],
                                       avg_dwellings), axis=1
    )
    features['No_Of_Dwellings'] = features['No_Of_Dwellings'].astype(int)

    # impute missing values for year built
    avg_year = features[features['Yr_Built'].notna()].groupby(
        ['Building_Class', 'TaxRate', 'Sq_Ft']
    ).agg(
        {'Yr_Built': 'mean'}
    ).reset_index()
    avg_year_class = features[features['Yr_Built'].notna()].groupby(
        ['Building_Class', 'Sq_Ft']
    ).agg(
        {'Yr_Built': 'mean'}
    ).reset_index()
    features['Yr_Built'] = features.apply(
        lambda x: fill_year_built(x['Building_Class'], x['TaxRate'],
                                  x['Sq_Ft'], x['Yr_Built'],
                                  avg_year, avg_year_class), axis=1
    )
    features['Yr_Built'] = features['Yr_Built'].astype(int)

    # impute missing values for recorded taxes
    avg_taxes = features[features['Recorded_Taxes'].notna()].groupby(
        ['Calculated_Taxes', 'Sq_Ft']
    ).agg(
        {'Recorded_Taxes': 'mean'}
    ).reset_index()
    features['Recorded_Taxes'] = features.apply(
        lambda x: fill_recorded_taxes(x['Calculated_Taxes'], x['Sq_Ft'],
                                      x['Recorded_Taxes'], avg_taxes), axis=1
    )
    
    # drop features we no longer need
    features = features.drop(['Calculated_Taxes', 'Calculated_Taxes_Year'], axis=1)

    # impute missing values for corporate owned housing
    avg_corp_owned = features[features['Corporate_Owned'].notna()].groupby(
        ['Absentee', 'No_Of_Dwellings', 'Building_Class']
    ).agg(
        {'Corporate_Owned': 'mean'}
    ).reset_index()
    avg_corp_owned_dwell = features[features['Corporate_Owned'].notna()].groupby(
        ['Absentee', 'No_Of_Dwellings']
    ).agg(
        {'Corporate_Owned': 'mean'}
    ).reset_index()
    avg_corp_owned_absent = features[features['Corporate_Owned'].notna()].groupby(
        ['Absentee']
    ).agg(
        {'Corporate_Owned': 'mean'}
    ).reset_index()
    features['Corporate_Owned'] = features.apply(
        lambda x: fill_corporate_owned(x['Absentee'], x['No_Of_Dwellings'],
                                       x['Building_Class'], x['Corporate_Owned'],
                                       avg_corp_owned, avg_corp_owned_dwell,
                                       avg_corp_owned_absent), axis=1
    )
    
    # convert values to numerical type
    features['Year_1'] = features['Year_1'].astype(int)
    return features


def convert_building_class(value):
    if pd.notna(value):
        if value.isdigit():
            return int(value)
        else:
            try:
                value = value.replace('+', '.5')
                return round(float(value))
            except ValueError:
                return None


def fill_building_class(tot_asmt, bld_cls, avg_bld_class):
    if (pd.isna(bld_cls)):
        valid_indices = avg_bld_class.index[avg_bld_class.index <= tot_asmt]
        if not valid_indices.empty:
          return avg_bld_class[valid_indices.max()]
        valid_indices = avg_bld_class.index[avg_bld_class.index > tot_asmt]
        if not valid_indices.empty:
          return avg_bld_class[valid_indices.min()]
        else:
          return 0
    else:
        return bld_cls


def fill_square_feet(bld_asmt, sq_ft, avg_feet):
    if (pd.isna(sq_ft)):
        valid_indices = avg_feet.index[avg_feet.index <= bld_asmt]
        if not valid_indices.empty:
          return avg_feet[valid_indices.max()]
        valid_indices = avg_feet.index[avg_feet.index > bld_asmt]
        if not valid_indices.empty:
          return avg_feet[valid_indices.min()]
        else:
          return 0
    else:
        return sq_ft


def fill_no_of_dwellings(bld_cls, sq_ft, no_dwell, avg_dwellings):
    if np.isnan(no_dwell):
        result = 0
        segment = avg_dwellings[(avg_dwellings['Building_Class'] == bld_cls)]     
        if not segment.empty:
            filtered = segment[segment['Sq_Ft'] <= sq_ft]
            if not filtered.empty:
                result = filtered.loc[filtered.idxmax()['Sq_Ft']]['No_Of_Dwellings']
            else:
                filtered = segment[segment['Sq_Ft'] > sq_ft]
                if not filtered.empty:
                    result = filtered.loc[filtered.idxmin()['Sq_Ft']]['No_Of_Dwellings']
        return round(result)
    else:
        return no_dwell


def fill_year_built(bld_cls, tax_rate, sq_ft, yr_built, avg_year, avg_year_class):
    if np.isnan(yr_built):
        result = 0
        segment = avg_year[(avg_year['Building_Class'] == bld_cls) &
                           (avg_year['TaxRate'] == tax_rate)]
        if segment.empty:
            segment = avg_year_class[avg_year_class['Building_Class'] == bld_cls]        
        if not segment.empty:
            filtered = segment[segment['Sq_Ft'] <= sq_ft]
            if not filtered.empty:
                result = filtered.loc[filtered.idxmax()['Sq_Ft']]['Yr_Built']
            else:
                filtered = segment[segment['Sq_Ft'] > sq_ft]
                if not filtered.empty:
                    result = filtered.loc[filtered.idxmin()['Sq_Ft']]['Yr_Built']
        return round(result)
    else:
        return yr_built


def fill_recorded_taxes(calc_tax, sq_ft, rec_tax, avg_taxes):
    if np.isnan(rec_tax):
        result = calc_tax
        segment = avg_taxes[avg_taxes['Calculated_Taxes'] == calc_tax]
        if segment.empty:
            filtered = avg_taxes[avg_taxes['Calculated_Taxes'] <= calc_tax]
            if not filtered.empty:
                calc_tax = filtered.loc[filtered.idxmax()['Calculated_Taxes']]['Calculated_Taxes']
                segment = avg_taxes[avg_taxes['Calculated_Taxes'] == calc_tax]
            else:
                filtered = avg_taxes[avg_taxes['Calculated_Taxes'] > calc_tax]
                if not filtered.empty:
                    calc_tax = filtered.loc[filtered.idxmin()['Calculated_Taxes']]['Calculated_Taxes']
                    segment = avg_taxes[avg_taxes['Calculated_Taxes'] == calc_tax]    
        if not segment.empty:
            filtered = segment[segment['Sq_Ft'] <= sq_ft]
            if not filtered.empty:
                result = filtered.loc[filtered.idxmax()['Sq_Ft']]['Recorded_Taxes']
            else:
                filtered = segment[segment['Sq_Ft'] > sq_ft]
                if not filtered.empty:
                    result = filtered.loc[filtered.idxmin()['Sq_Ft']]['Recorded_Taxes']
        return result
    else:
        return rec_tax


def fill_corporate_owned(absentee, no_dwell, bld_cls, corp_owned,
                         avg_corp_owned, avg_corp_owned_dwell, avg_corp_owned_absent):
    if pd.isnull(corp_owned):
        result = 0
        segment = avg_corp_owned[(avg_corp_owned['Absentee'] == absentee) &
                                 (avg_corp_owned['No_Of_Dwellings'] == no_dwell) &
                                 (avg_corp_owned['Building_Class'] == bld_cls)]
        if segment.empty:
            segment = avg_corp_owned_dwell[(avg_corp_owned_dwell['Absentee'] == absentee) &
                                           (avg_corp_owned_dwell['No_Of_Dwellings'] == no_dwell)]
        if segment.empty:
            segment = avg_corp_owned_absent[(avg_corp_owned_absent['Absentee'] == absentee)]  
        if not segment.empty:
            result = segment.mean()['Corporate_Owned']
        return round(result)
    else:
        return corp_owned
