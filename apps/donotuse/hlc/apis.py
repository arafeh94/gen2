from src.federated.events import FederatedSubscriber

def get_nearly_equal_numbers(x, n):
    # Divide x into n nearly equal parts
    quotient, remainder = divmod(x, n)
    numbers = [quotient] * n

    # Adjust the last number to make the sum equal to x
    for i in range(remainder):
        numbers[i] += 1

    return numbers

class SaveClientsModels(FederatedSubscriber):
    def __init__(self):
        super().__init__()
        self.clients_weights = {}
        self.sample_size = {}

    def on_trainer_end(self, params):
        context, id, weights, sample_size = params['context'], params['trainer_id'], params['weights'], params[
            'sample_size']
        self.clients_weights[id] = weights
        self.sample_size[id] = sample_size
