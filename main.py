import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import tensorboard

from keras import Sequential
from sklearn.preprocessing import StandardScaler
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def plot_learning_curves(history, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def parse_input():
    arg_list = []
    for arg in sys.argv[1:]:
        arg_list.append(float(arg))
    return np.array(arg_list).reshape(1, -1)

def predict_with_trained_model(model):
    user_input = parse_input()
    prediction = model.predict(user_input)
    print(f"Prediction: {prediction}")
    predicted_class = np.argmax(prediction, axis=1)
    print("Predicted Class:", predicted_class+1)

def wrangle_data(path):
    scaler = StandardScaler()
    np.random.seed(42)
    tf.random.set_seed(42)

    # Basic data wrangling
    wine_array = np.genfromtxt('wine/wine.data', delimiter=',')
    np.random.shuffle(wine_array)
    labels = wine_array[:, 0].astype(int)
    features = wine_array[:, 1:]
    features = scaler.fit_transform(features)

    one_hot_labels = np.zeros((labels.size, labels.max()))
    one_hot_labels[np.arange(labels.size), labels - 1] = 1

    wine_array = np.concatenate((features, one_hot_labels),axis=1)

    ## Dataset splitting
    wine_array_train, wine_array_test = train_test_split(wine_array, test_size=0.1, random_state=42)
    wine_array_train, wine_array_val = train_test_split(wine_array, test_size=0.1, random_state=42)

    x_train = wine_array_train[:, :-3]
    y_train = wine_array_train[:, -3:]

    x_test = wine_array_test[:, :-3]
    y_test = wine_array_test[:, -3:]

    x_val = wine_array_val[:, :-3]
    y_val = wine_array_val[:, -3:]

    return x_train, y_train, x_test, y_test, x_val, y_val

def define_model(reluLayers, neuronPerLayer, neuronNumberChange = "flat"):
    model = Sequential()
    model.add(tf.keras.layers.Dense(13, input_shape=(13,), activation='relu')) # Input layer stays the same
    model.add(tf.keras.layers.Dropout(0.2))
    if neuronNumberChange == "flat":
        for i in range(int(reluLayers)):
            model.add(tf.keras.layers.Dense(neuronPerLayer, activation='relu'))
    elif neuronNumberChange == "ascending":
        for i in range(int(reluLayers)):
            model.add(tf.keras.layers.Dense(int(i*len(reluLayers))*neuronPerLayer, activation='relu'))
    elif neuronNumberChange == "descending":
        for i in range(int(reluLayers)):
            model.add(tf.keras.layers.Dense(int((reluLayers-i)*len(reluLayers))*neuronPerLayer, activation='relu'))

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, x_val, y_val = wrangle_data('wine/train')

    models_array = []
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/fit", histogram_freq=1)
    maxLayersNumber = 4
    maxNeuronsNumber = 8
    for layers in range(1,maxLayersNumber):
        for neurons in range(1,maxNeuronsNumber):
            models_array.append(define_model(layers, neurons))

    input_data = tf.random.normal((10, 13))  # 10 samples, each with 13 features
    target_data = tf.random.normal((10, 3))  # 10 samples, each with 3 output classes (softmax)

    # Train the models and visualize with TensorBoard
    for model in models_array:
        model.summary()  # Print model summary

    exit()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

    print("Trenowanie modelu Standard...")
    history_standard = model_standard.fit(
        x_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.3,
        shuffle=False,
        callbacks=[early_stopping]
    )

    print("Trenowanie modelu Powiększonego...")
    history_bigger = model_bigger.fit(
        x_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.3,
        shuffle=False,
    )

    plot_learning_curves(history_standard, "Standard Model")
    plot_learning_curves(history_bigger, "Bigger Model")

    print(f'Test Accuracy (standard): {history_standard.accuracy:.2f}')
    print(f'Test Accuracy (bigger): {history_standard.accuracy_bigger:.2f}')

    predict_with_trained_model()