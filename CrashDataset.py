import os
from deepproblog.dataset import ImageDataset
from deepproblog.query import Query
from problog.logic import Term, Constant
import torchvision.transforms as transforms


path = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class crashDataset(ImageDataset):

    def __init__(
        self,
        subset,
    ):
        super().__init__("{}/data/{}/images/".format(path, subset))
        self.subset = subset
        self.data = []
        with open("{}/data/{}/labels.csv".format(path, subset)) as f:
            for line in f:
                self.data.append(line)

    def to_query(self, i):
        l = Constant(self.data[i])
        return Query(
            Term("car_obstruction_distance", l),
            substitution={Term("a"): Constant(i)})

    def __getitem__(self, i):
        return super().__getitem__("out{}".format(i))

    def __len__(self):
        return len(self.data)


train = crashDataset("train")
test = crashDataset("test")
