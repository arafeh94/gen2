from src.apis.extensions import Dict
from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning


def integrate(context, acc):
    for index, ac in enumerate(acc):
        if index not in context.history:
            context.history[index] = Dict()
        context.history[index].update({'pre_acc': ac})


class PreAccIntegrator(FederatedSubscriber):
    def __init__(self, acc):
        super().__init__()
        self.acc = acc

    def on_federated_started(self, params):
        context = params['context']
        integrate(context, self.acc)
