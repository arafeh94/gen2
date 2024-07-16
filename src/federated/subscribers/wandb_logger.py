import atexit

from src.federated.events import FederatedSubscriber
from src.federated.federated import FederatedLearning
from src.manifest import wandb_config


class WandbLogger(FederatedSubscriber):
    def __init__(self, config=None, resume=False, id: str = None):
        super().__init__()
        import wandb
        wandb.login(key=wandb_config['key'])
        self.wandb = wandb
        self.config = config
        self.id = id
        self.resume = resume
        atexit.register(lambda: self.wandb.finish())

    def on_init(self, params):
        if self.resume:
            if self.id is None:
                raise Exception('resumable requires that id is not None')
        resume = 'allow' if self.resume else None
        self.wandb.init(project=wandb_config['project'], entity=wandb_config['entity'], config=self.config,
                        id=self.id, resume=resume)

    def on_round_end(self, params):
        context: FederatedLearning.Context = params['context']
        self.wandb.log(context.last_entry())

    def on_federated_ended(self, params):
        self.wandb.finish()
