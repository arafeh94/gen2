import json
import sys

sys.path.append("../../")

from src import manifest
from src.data.data_provider import PickleDataProvider

url = sys.argv[1]
dc = PickleDataProvider(manifest.datasets_urls[url]).collect()
