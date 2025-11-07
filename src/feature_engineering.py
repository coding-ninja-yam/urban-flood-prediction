def add_features(df):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    df = df.sort_values(['station_id', 'date'])

    # Rolling rainfall
    df['rain_past3'] = df.groupby('station_id')['RR'].rolling(3).sum().shift(1).reset_index(0, drop=True)
    df['rain_past7'] = df.groupby('station_id')['RR'].rolling(7).sum().shift(1).reset_index(0, drop=True)

    # Humidity & temperature trends
    df['RH_trend3'] = df.groupby('station_id')['RH_avg'].rolling(3).mean().shift(1).reset_index(0, drop=True)
    df['Tavg_trend3'] = df.groupby('station_id')['Tavg'].rolling(3).mean().shift(1).reset_index(0, drop=True)

    # Rain intensity (z-score)
    df['rain_intensity'] = df.groupby('station_id')['RR'].transform(lambda x: (x - x.mean()) / x.std())

    df = df.fillna(0)
    return df

