import pandas as pd
from sklearn.preprocessing import RobustScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Amount'] = RobustScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
    return df

def split_data(df, train_frac=0.8, val_frac=0.1):
    df = df.sample(frac=1, random_state=1)
    train_end = int(train_frac * len(df))
    val_end = int(val_frac * len(df)) + train_end
    train, val, test = df[:train_end], df[train_end:val_end], df[val_end:]
    return train, val, test
