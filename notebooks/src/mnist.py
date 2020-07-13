import tensorflow as tf

Sequential = tf.keras.Sequential
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
np_utils = tf.keras.utils.to_categorical


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils(y_train)
y_test = np_utils(y_test)
num_classes = y_test.shape[1]

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(
        Dense(
            num_pixels,
            input_dim=num_pixels,
            kernel_initializer='normal',
            activation='relu',
        )
    )
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
    )
    return model


def main():
    # build the model
    model = baseline_model()
    # Fit the model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=200,
        verbose=2,
    )
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))


if __name__ == "__main__":
    main()
