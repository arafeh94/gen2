import torch
from torch import nn

from src.data.data_container import DataContainer
from src.federated.protocols import Trainer, ModelInfer
from sklearn.metrics import precision_recall_fscore_support


class AccLoss(ModelInfer):
    def __init__(self, batch_size: int, criterion, device=None):
        super().__init__(batch_size, criterion)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, model: nn.Module, test_data: DataContainer):
        model.to(self.device)
        model.eval()
        test_loss = test_acc = test_total = 0.
        criterion = self.criterion
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data.batch(self.batch_size)):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
        acc, loss = test_acc / test_total, test_loss / test_total
        return {'acc': acc, 'loss': loss}


class F1(ModelInfer):
    def __init__(self, batch_size: int, criterion, device=None):
        super().__init__(batch_size, criterion)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def infer(self, model: nn.Module, test_data: DataContainer):
        model.to(self.device)
        model.eval()

        test_loss = test_acc = test_total = 0.
        all_targets = []
        all_predictions = []
        criterion = self.criterion

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data.batch(self.batch_size)):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
                all_targets.extend(target.cpu().tolist())
                all_predictions.extend(predicted.cpu().tolist())

        acc = test_acc / test_total
        loss = test_loss / test_total
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted',
                                                                   zero_division=0)

        return {'acc': acc, 'loss': loss, 'precision': precision, 'recall': recall, 'f1': f1}
