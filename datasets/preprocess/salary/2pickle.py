import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer, MinMaxScaler

from src.data.data_container import DataContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

path = "./adult.csv"

df = pd.read_csv(path)
encoded_dataframe = df.copy()
encoder = LabelEncoder()

for column in ['workclass', 'education', 'income', 'native-country', 'marital-status', 'occupation', 'relationship',
               'race', 'gender']:
    encoded_dataframe[column] = encoder.fit_transform(encoded_dataframe[column])

normalizer = Normalizer(norm='l2')
column_to_normalize = df['fnlwgt'].values.reshape(1, -1)
normalized = normalizer.fit_transform(column_to_normalize)
encoded_dataframe['fnlwgt'] = normalized[0]

print(encoded_dataframe)
x = []
y = []
for index, row in encoded_dataframe.iterrows():
    xr = row.values[0:-1].tolist()
    yr = row.values[-1].tolist()
    x.append(xr)
    y.append(yr)

container = DataContainer(np.array(x), np.array(y))
pickle.dump(container, open('salary.pkl', 'wb'))
