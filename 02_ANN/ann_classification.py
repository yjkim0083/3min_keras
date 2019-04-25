# 분류 ANN을 위한 인공지능 모델 구현
from keras import layers, models

# 분산 방식 모델링을 포함하는 함수형 구현
def ANN_models_func(Nin, Nh, Nout):
    x = layers.Input(shape=(Nin,))
    h = layers.Activation("relu")(layers.Dense(Nh)(x))
    y = layers.Activation("softmax")(layers.Dense(Nout)(h))
    model = models.Model(x, y)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# 연쇄 방식 모델링을 포함하는 함수형 구현
def ANN_seq_func(Nin, Nh, Nout):
    model = models.Sequential()
    model.add(layers.Dense(Nh, activation="relu", input_shape=(Nin,)))
    model.add(layers.Dense(Nout, activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model

# 분산 방식 모델링을 포함하는 객체지향형 구현
class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        # Prepare network layers and activate functions
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation("relu")
        softmax = layers.Activation("softmax")

        # Connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)

        self.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

# 연쇄 방식 모델링을 포함하는 객체지향형 구현
class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        self.add(layers.Dense(Nh, activation="relu", input_shape=(Nin,)))
        self.add(layers.Dense(Nout, activation="softmax"))
        self.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])


# 분류 ANN에 사용할 데이터 불러오기
import numpy as np
from keras import datasets
from keras.utils import np_utils

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical((y_test))

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

# 분류 ANN 학습결과 그래프 구현
import matplotlib.pyplot as plt

def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc=0)

def plot_acc(history):
    # summarize history for accuracy
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train","Test"], loc=0)

# 분류 ANN 학습 및 성능 분석
def main():
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class

    model = ANN_seq_class(Nin, Nh, Nout)
    (X_train, Y_train), (X_test, Y_test) = Data_func()

    ################################
    # Training
    ################################
    history = model.fit(X_train,
                        Y_train,
                        epochs=15,
                        batch_size=100,
                        validation_split=0.2)

    performance_test = model.evaluate(X_test,
                                      Y_test,
                                      batch_size=100)

    print("Test Loss and Accuracy ->", performance_test)

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()

# Run code
if __name__ == "__main__":
    main()