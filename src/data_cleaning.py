import pandas as pd

def clean_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df = df.sort_values(by=['station_id', 'date']).reset_index(drop=True)

    # Fill missing values
    df['RR'] = df['RR'].fillna(0)
    df['ddd_car'] = df['ddd_car'].ffill()

    num_cols = ['Tn','Tx','Tavg','RH_avg','ss','ff_x','ddd_x','ff_avg']
    for col in num_cols:
        df[col] = df.groupby('station_id')[col].transform(lambda x: x.fillna(x.median()))

    return df

