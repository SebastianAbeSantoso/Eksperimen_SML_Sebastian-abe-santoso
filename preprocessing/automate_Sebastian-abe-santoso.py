import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(raw_path, output_path):
    df = pd.read_csv(raw_path)

    if str(df.iloc[0, 0]).startswith('#'):
        df = df.iloc[1:].reset_index(drop=True)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['usdprice'] = pd.to_numeric(df['usdprice'], errors='coerce')

    df = df[df['unit'] == 'KG'].reset_index(drop=True)
    df = df.dropna(subset=['admin1', 'admin2', 'latitude', 'longitude']).reset_index(drop=True)
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)

    df = df[['date', 'admin1', 'market', 'category', 'commodity', 'price']].copy()
    df.columns = ['date', 'province', 'market', 'category', 'commodity', 'price']

    df['year_month'] = df['date'].dt.to_period('M')
    monthly = (
        df.groupby(['province', 'market', 'commodity', 'year_month'])['price']
        .mean()
        .reset_index()
    )
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly.sort_values(['province', 'market', 'commodity', 'year_month']).reset_index(drop=True)

    group_cols = ['province', 'market', 'commodity']
    monthly['lag_1'] = monthly.groupby(group_cols)['price'].shift(1)
    monthly['lag_2'] = monthly.groupby(group_cols)['price'].shift(2)
    monthly['lag_3'] = monthly.groupby(group_cols)['price'].shift(3)
    monthly['rolling_avg_3'] = monthly.groupby(group_cols)['price'].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )
    monthly['month'] = monthly['year_month'].dt.month
    monthly['year'] = monthly['year_month'].dt.year
    monthly['target'] = monthly.groupby(group_cols)['price'].shift(-1)

    model_df = monthly.dropna(
        subset=['lag_1', 'lag_2', 'lag_3', 'rolling_avg_3', 'target']
    ).reset_index(drop=True)

    le_province = LabelEncoder()
    le_market = LabelEncoder()
    le_commodity = LabelEncoder()
    model_df['province_enc'] = le_province.fit_transform(model_df['province'])
    model_df['market_enc'] = le_market.fit_transform(model_df['market'])
    model_df['commodity_enc'] = le_commodity.fit_transform(model_df['commodity'])

    model_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Shape: {model_df.shape}")
    return model_df


if __name__ == "__main__":
    RAW_PATH = "wfpfoodpricesidn_raw.csv"
    OUTPUT_PATH = "preprocessing/wfpfoodpricesidn_preprocessing.csv"
    preprocess_data(RAW_PATH, OUTPUT_PATH)