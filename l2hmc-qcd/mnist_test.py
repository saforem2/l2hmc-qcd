import tensorflow as tf


def main():
     mnist = tf.keras.datasets.mnist
     (x_train, y_train), (x_test, y_test) = mnist.load_data()
     x_train, x_test = x_train / 255.0, x_test / 255.0
     model = tf.keras.models.Sequential([
         tf.keras.layers.Flatten(input_shape=(28, 28)),
         tf.keras.layers.Dense(128, activation='relu'),
         tf.keras.layers.Dropout(0.1),
         tf.keras.layers.Dense(10)
     ])
     predictions = model(x_train[:1]).numpy()
     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
     model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
     history = model.fit(x_train, y_train, epochs=5)
     eval_results = model.evaluate(x_test, y_test, verbose=2)
     print('Done!')


if __name__ == '__main__':
    main()
