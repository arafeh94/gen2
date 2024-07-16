import json

import h5py
import mysql.connector

from libs import language_tools
from src.data.data_container import DataContainer
from src.data.data_provider import PickleDataProvider

data = h5py.File('data.h5', 'r')
print(data)