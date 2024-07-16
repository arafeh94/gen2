from apps.fed_apps.WarmupApp import WarmupApp
from src.app.federated_app import FederatedApp
from src.app.settings import Settings
from src.federated.subscribers.analysis import ShowWeightDivergence
from src.federated.subscribers.fed_plots import EMDWeightDivergence

TEST_WARMUP = 1

if TEST_WARMUP:
    app = WarmupApp(Settings.from_json_file("mnist.json"), wp_ratio=0.2, batch=50, epochs=500)
else:
    app = FederatedApp(Settings.from_json_file("mnist.json"))

app.start(ShowWeightDivergence(plot_type='linear', caching=True), EMDWeightDivergence(show_plot=True))
