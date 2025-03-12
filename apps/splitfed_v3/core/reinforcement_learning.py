def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@singleton
class ReinforcementLearningEnv:
    def __init__(self, num_clients=100, num_clusters=3):
        self.num_clients = num_clients
        self.num_clusters = num_clusters

    def reset(self, client_data):
        pass

    def step(self, action, real_time_info):
        pass
