import copy
from typing import Tuple

from torch import nn
import tenseal as ts

from src.data.data_container import DataContainer
from src.federated.components.trainers import TorchTrainer
from src.federated.federated import FederatedLearning
from src.federated.protocols import TrainerParams


class EncryptionClient(TorchTrainer):

    def __init__(self):
        super().__init__()
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.generate_galois_keys()
        self.context.global_scale = 2 ** 40

    def train(self, model: nn.Module, train_data: DataContainer, context: FederatedLearning.Context,
              config: TrainerParams) -> Tuple[any, int]:
        # encrypt_data
        encrypted_data = self.encrypt(data=train_data)
        return super().train(model, encrypted_data, context, config)

    def encrypt(self, data):
        dc = DataContainer(ts.ckks_tensor(self.context, data.x), ts.ckks_tensor(self.context, data.y))
        print('data encrypted')
        return dc
