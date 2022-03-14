import torch
from deepproblog.engines import ExactEngine
from network import DANGERDIST_CNN
from deepproblog.network import Network
from deepproblog.model import Model
from CrashDataset import train
from CrashDataset import test
from deepproblog.dataset import DataLoader
from deepproblog.train import train_model


if __name__ == '__main__':
    network = DANGERDIST_CNN()
    net = Network(network, "dangerdist_cnn", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    model = Model("crash_detection.pl", [net])
    model.add_tensor_source("train", train)
    model.add_tensor_source("test", test)
    loader = DataLoader(train, 2, False)
    model.set_engine(ExactEngine(model), cache=True)
    trainObj = train_model(model, loader, 1, log_iter=100, profile=0)



