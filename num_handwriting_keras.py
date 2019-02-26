import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = keras.utils.normalize(x_test, axis=1)
model_name = 'epic_num_reader.model'
def training():

    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = keras.utils.normalize(x_train,axis=1)
    x_test = keras.utils.normalize(x_test,axis=1)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu',))
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(128,activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=3)

    val_loss, val_acc = model.evaluate(x_test,y_test)
    print(val_loss)
    print(val_acc)

    model.save(model_name)

def pridiction(model):
    new_model = keras.models.load_model(model_name)
    pridict = new_model.predict([x_test])
    print(np.argmax(pridict[1]))
    plt.imshow(x_test[0])
    plt.show()
if __name__ == '__main__':
    pridiction(model_name)
