# 완전 연결 계층 AE 모델링
from keras import layers, models

class AE(models.Model):
    def __init__(self, x_nodes=784, z_dim=36):
        x_shape = (x_nodes, )
        x = layers.Input(shape=x_shape)
        z = layers.Dense(z_dim, activation="relu")(x)
        y = layers.Dense(x_nodes, activation="sigmoid")(z)

        super().__init__(x, y)

        self.x = x
        self.z = z
        self.z_dim = z_dim

        self.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    def Encoder(self):
        return models.Model(self.x, self.z)

    def Decoder(self):
        z_shape = (self.z_dim, )
        z = layers.Input(shape=z_shape)
        y_layer = self.layers[-1]
        y = y_layer(z)
        return models.Model(z, y)

# 데이터 준비
from keras.datasets import mnist
import numpy as np

(X_train, _), (X_test, _) = mnist.load_data()

X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print("X_train.shape:", X_train.shape)
print("X_test.shape:",X_test.shape)

# 학습 효과 분석
from kerasapp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt

# 완전연결 계층 AE 동작 확인
def show_ae(autoencoder):
    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()

    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))

    for i in range(n):
        ax = plt.subplot(3, n, i+1)
        plt.imshow(X_test[i].reshape(28,28))
        plt.gray()

        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

# 학습 및 성능 평가
def main():
    x_nodes = 784
    z_dim = 36

    autoencoder = AE(x_nodes, z_dim)
    history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True, validation_data=(X_test, X_test))

    plot_acc(history)
    plt.show()
    plot_loss(history)
    plt.show()

    show_ae(autoencoder)
    plt.show()

if __name__ == "__main__":
    main()