import typing

from src.apis.extensions import TorchModel, Dict
from src.app.federated_app import FederatedApp
from src.app.session import Session
from src.app.settings import Settings
from src.data.data_container import DataContainer
from src.apis import lambdas


class WarmupApp(FederatedApp):
    def __init__(self, settings: Settings, **kwargs):
        """
        warmup args extended from FederatedApps
        Args:
            settings: load settings file from json or as dict
            wp_ratio: float, default to 0.05
            batch: int, default to 50
            epochs: int, default to 25
            **kwargs:
        """
        super().__init__(settings, **kwargs)

    def init_federated(self, session: Session):
        federated = super(WarmupApp, self).init_federated(session)
        client_data: Dict[int:DataContainer] = federated.trainers_data_dict
        wp_ratio = self.kwargs.get('wp_ratio', 0.05)
        wp_batch = self.kwargs.get('batch', 50)
        wp_epochs = self.kwargs.get('epochs', 25)
        w_d = client_data.map(lambda ci, dc: dc.shuffle(42).split(wp_ratio)[0]).reduce(lambdas.dict2dc).as_tensor()
        c_d = client_data.map(lambda ci, dc: dc.shuffle(42).split(wp_ratio)[1]).map(lambdas.as_tensor)
        federated.trainers_data_dict = c_d
        model = federated.initial_model() if callable(federated.initial_model) else federated.initial_model
        TorchModel(model).train(w_d.batch(wp_batch), epochs=wp_epochs, lr=federated.trainer_config.args.get('lr'),
                                optimizer=federated.trainer_config.get_optimizer()(model),
                                criterion=federated.trainer_config.get_criterion())
        return federated


if __name__ == '__main__':
    settings = Settings.from_json_file('../experiments/federated_application/c_warmup.json')
    app = WarmupApp(settings, wp_ratio=0.05, batch=50, epochs=50)
    app.start()
