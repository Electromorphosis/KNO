# KNO Lab 6

## Zadanie wstępne
Wynik po ostatniej epoce:
```
Epoch 6/6
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 0.0590 - sparse_categorical_accuracy: 0.9834 - val_loss: 0.0762 - val_sparse_categorical_accuracy: 0.9774
```
dla tensorflow-datasets 4.9.7 (najnowsza)

```
Epoch 6/6
469/469 ━━━━━━━━━━━━━━━━━━━━ 1s 1ms/step - loss: 0.0589 - sparse_categorical_accuracy: 0.9833 - val_loss: 0.0811 - val_sparse_categorical_accuracy: 0.9757
```
dla tensorflow-datasets 4.9.2

## Zadanie 1
Sieć konwolucyjna:
```
model_convolutional = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10)
])
```

Wynik dla niej:
```
Epoch 6/6
469/469 ━━━━━━━━━━━━━━━━━━━━ 9s 18ms/step - loss: 0.0226 - sparse_categorical_accuracy: 0.9928 - val_loss: 0.0316 - val_sparse_categorical_accuracy: 0.9904
```