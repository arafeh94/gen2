from src.apis.extensions import Dict
from src.federated.protocols import ClientScanner


class DefaultScanner(ClientScanner):
    def __init__(self, client_data: Dict):
        self.client_data = client_data

    def scan(self):
        return self.client_data


class PositionScanner(ClientScanner):
    def __init__(self, count: int):
        self.count = count

    def scan(self):
        cnt = {}
        for i in range(self.count):
            cnt[i] = None
        return cnt
