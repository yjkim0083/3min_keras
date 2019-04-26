# 분류 CNN의 학습 및 성능 평가 수행
from keras import datasets
import keras
assert keras.backend.image_data_format() == "channels_last"

from kerasapp import aicnn

class Machine(aicnn.Machine):
    def __init__(self):
        (X, y), (x_test, y_test) = datasets.cifar10.load_data()
        super().__init__(X, y, nb_classes=10)


def main():
    m = Machine()
    m.run()

if __name__ == "__main__":
    main()