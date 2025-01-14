import pandas as pd
from sklearn.preprocessing import RobustScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Normalize 'Amount' and scale 'Time'
    df['Amount'] = RobustScaler().fit_transform(df['Amount'].to_numpy().reshape(-1, 1))
    time = df['Time']
    df['Time'] = (time - time.min()) / (time.max() - time.min())
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=1)
    return df

def split_data(df):
    train = df[:240000]
    test = df[240000:262000]
    val = df[262000:]
    return train, test, val
