class ExecutionTimeModel:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ExecutionTimeModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.models = {
            "elow": pickle.load(open(self._path('elow_gp_model.pkl'), 'rb')),
            "low": pickle.load(open(self._path('low_svr_model.pkl'), 'rb')),
            "medium": pickle.load(open(self._path('medium_svr_model.pkl'), 'rb')),
            "high": pickle.load(open(self._path('high_svr_model.pkl'), 'rb')),
            "scaler": pickle.load(open(self._path('svr_scaler.pkl'), 'rb')),
            "t1": pickle.load(open(self._path('t1_svr_model.pkl'), 'rb')),
        }

        self.speed_map = {
            't1': [0.1, 0.25, 1]
            # "elow": [0.01, 0.025, 0.05],
            # "low": [0.1, 0.25, 0.5],
            # "medium": [1, 1.5, 2, 2.5],
            # "high": [3, 4, 5],
        }

    def get_model(self, val):
        used_model = self.get_speed(val)
        return self.models[used_model]

    def get_speed(self, val):
        for speed_name, vals in self.speed_map.items():
            if val in vals:
                return speed_name
        raise Exception("please use only the following speeds: {}".format(self.speed_map))

    def _path(self, item):
        try1 = './files/{}'.format(item)
        if os.path.exists(try1):
            return try1
        paths = [item, 'files', './']
        path = ''
        for e_path in paths:
            path = '/' + e_path + path
            if os.path.exists(path):
                return path
        raise Exception("Can't find model")

    def predict(self, ram, cpu, disk, latency, speed) -> float:
        features = [[ram, cpu, disk, latency]]
        if self.get_speed(speed) != 'elow':
            scaled = self.models['scaler'].transform(features)
            return self.get_model(speed).predict(scaled)[0]
        return self.get_model(speed).predict(features)[0]

    def predict2(self, client: Client):
        ram, cpu, disk, latency = client.configs['ram_a'], client.configs['cpu_a'], client.configs['disk_a'], \
            client.configs['latency']
        return self.predict(ram, cpu, disk, latency, client.speed)