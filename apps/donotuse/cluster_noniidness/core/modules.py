from src.federated.components.trainer_manager import SeqTrainerManager
from src.federated.protocols import TrainerParams, Trainer


class Cluster_TrainerProvider(SeqTrainerManager.TrainerProvider):
    def __init__(self, cluster_trainers):
        self.cluster_trainers = cluster_trainers

    def collect(self, trainer_id, config: TrainerParams) -> Trainer:
        return self.cluster_trainers[trainer_id]
