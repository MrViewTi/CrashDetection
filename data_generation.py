from PIL import Image
from PIL import ImageOps
import numpy as np
import random


def generate(size):
    data = np.ndarray((size, 40, 200), dtype=float)
    labels = np.ndarray(size, dtype=float)
    road = Image.open('road.png')
    car = Image.open('car.png')
    obs = Image.open('obs.png')
    y = (road.size[1] - car.size[1]) // 2
    for i in range(size):
        img = road.copy()
        x1 = int(random.random() * road.size[0])
        while x1 >= (road.size[0] - car.size[0] - obs.size[0]):
            x1 = int(random.random() * road.size[0])
        img.paste(car, (x1, y))
        x2 = int(random.random() * road.size[0])
        while x2 < (x1 + car.size[0]) or x2 >= road.size[0] - obs.size[0]:
            x2 = int(random.random() * road.size[0])
        img.paste(obs, (x2, y))
        img.save('data\out' + str(i) + '.png')
        data[i] = np.asarray(img, dtype=float)
        labels[i] = float(x2 - (x1 + car.size[0]) < 30)
    np.save('data\data', data)
    np.save('data\labels', labels)
