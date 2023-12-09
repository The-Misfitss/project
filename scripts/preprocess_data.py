import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def normalize(df):
    # Define the columns to normalize
    columns_to_normalize = ['Reading']
    # Apply Min-Max normalization
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    df[columns_to_normalize] = df[columns_to_normalize].apply(lambda x: round(x, 3))
    # Convert 'Machine_ID' and 'Sensor_ID' to numeric values
    label_encoder = LabelEncoder()
    df['Machine_ID'] = label_encoder.fit_transform(df['Machine_ID'])
    df['Sensor_ID'] = label_encoder.fit_transform(df['Sensor_ID'])
    
    df['Machine_ID'] = df['Machine_ID'].fillna(df['Machine_ID'].mean())
    df['Sensor_ID'] = df['Sensor_ID'].fillna(df['Sensor_ID'].mean())
    df['Reading'] = df['Reading'].fillna(df['Reading'].mean())
    df['Timestamp'] = df['Timestamp'].ffill()
    
    # BREAK THE TIME STAMP INTO SEPARATE COLUMNS
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    
    cols = df.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    df = df[cols]
    
    # df = df.drop(columns=['Timestamp'])
    return df
def perform_train_test_split(df):
    # Split the data into training and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    # Save the data
    cwd = os.getcwd()
    train.to_csv(os.path.join(cwd,'data/processed/train/train.csv'), index=False)
    test.to_csv(os.path.join(cwd,'data/processed/test/test.csv'), index=False)

if __name__ == "__main__":
    # Assuming your data is in a CSV file named 'sensor_data.csv'
    file_path = 'data/raw/sensor_data.csv'
    file_path = os.path.join(os.getcwd(), file_path)
    df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    df = normalize(df)
    perform_train_test_split(df)
    print(df.head())