
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# Definindo e quantificando imagens de treino e validação
dataset_dir = os.path.join(os.getcwd(), "archive")

dataset_train_dir = os.path.join(dataset_dir, "training_data")
dataset_train_cracked_len = len(os.listdir(
    os.path.join(dataset_train_dir, "cracked")))
dataset_train_normal_len = len(os.listdir(
    os.path.join(dataset_train_dir, "normal")))

dataset_validation_dir = os.path.join(dataset_dir, "testing_data")
dataset_validation_cracked_len = len(os.listdir(
    os.path.join(dataset_validation_dir, "cracked")))
dataset_validation_normal_len = len(os.listdir(
    os.path.join(dataset_validation_dir, "normal")))

# print("Train Cracked: %s" % dataset_train_cracked_len)
# print("Train Normal: %s" % dataset_train_normal_len)

# print("Validation Cracked: %s" % dataset_validation_cracked_len)
# print("Validation Normal: %s" % dataset_validation_normal_len)

# Configurações de imagem
image_width = 416
image_height = 416
image_color_channel = 3
image_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

# Configurações de remessas de treinamento
batch_size = 32
epochs = 20
learning_rate = 0.0001

class_name = ["cracked",  "normal"]

# Configurações de treinamento
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

# Configurações de validação
dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_validation_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

dataset_validation_cardinality = tf.data.experimental.cardinality(
    dataset_validation)
dataset_validation_batches = dataset_validation_cardinality // 5

dataset_test = dataset_validation.take(dataset_validation_batches)
dataset_validation = dataset_validation.skip(dataset_validation_batches)

# print("Validation Dataset Cardinality: %d" % tf.data.experimental.cardinality(dataset_validation))
# print("Test Dataset Cardinality: %d " %tf.data.experimental.cardinality(dataset_test))


autotune = tf.data.AUTOTUNE

dataset_train = dataset_train.prefetch(buffer_size = autotune)
dataset_validation = dataset_validation.prefetch(buffer_size = autotune)
dataset_test = dataset_validation.prefetch(buffer_size = autotune)

# def plot_dataset(dataset):

#     plt.gcf().clear()
#     plt.figure(figsize = (15, 15))

#     for features, labels in dataset.take(1):

#         for i in range(9):

#             plt.subplot(3, 3, i + 1)
#             plt.axis('off')

#             plt.imshow(features[i].numpy().astype('uint8'))
#             plt.title(class_name[labels[i]])

#
# plot_dataset(dataset_train)

#
# plot_dataset(dataset_validation)

#
# plot_dataset(dataset_test)
# ### DATA AUGMENTATION

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])


def plot_dataset_data_augmentation(dataset):

    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for features, _ in dataset.take(1):

        feature = features[0]

        for i in range(9):

            feature_data_augmentation = data_augmentation(tf.expand_dims(feature, 0))

            plt.subplot(3, 3, i + 1)
            plt.axis('off')

            plt.imshow(feature_data_augmentation[0] / image_color_channel_size)

plot_dataset_data_augmentation(dataset_train)
# ### LAYER RESCALING

# Assuming you have defined image_shape and image_color_channel_size variables

# Define input layer with the specified shape
input_layer = tf.keras.Input(shape=image_shape)

# Rescaling layer
rescaling = tf.keras.layers.Rescaling(1. / (image_color_channel_size / 2.), offset=-1)(input_layer)

# After this, you can continue building your model
# ### TRANSFER LEARNING

# Load MobileNetV2 with ImageNet weights
model_transfer_learning = tf.keras.applications.MobileNetV2(input_shape=image_shape, include_top=False, weights='imagenet')

# Freeze the pre-trained weights
model_transfer_learning.trainable = False

# Display model summary
model_transfer_learning.summary()
# ### EARLY STOPPING

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
# ### MODEL

# Define your rescaling layer
rescaling = tf.keras.layers.Rescaling(1. / image_color_channel_size / 2 , offset = -1, input_shape = image_shape)

# Define your data augmentation layer
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# Assuming you have defined model_transfer_learning

# Define your model
model = tf.keras.models.Sequential([
    rescaling,
    data_augmentation,
    model_transfer_learning,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
learning_rate = 0.001  # Example learning rate, adjust as needed
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# Display model summary
model.summary()

history = model.fit(
    dataset_train,
    validation_data = dataset_validation,
    epochs = epochs,
    callbacks = [
        early_stopping
    ]
)

def plot_model():

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.gcf().clear()
    plt.figure(figsize = (15, 8))

    plt.subplot(1, 2, 1)
    plt.title('Training and Validation Accuracy')
    plt.plot(epochs_range, accuracy, label = 'Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label = 'Validation Accuracy')
    plt.legend(loc = 'lower right')

    plt.subplot(1, 2, 2)
    plt.title('Training and Validation Loss')
    plt.plot(epochs_range, loss, label = 'Training Loss')
    plt.plot(epochs_range, val_loss, label = 'Validation Loss')
    plt.legend(loc = 'lower right')

    plt.show()

plot_model()
# ### MODEL EVALUATION

dataset_test_loss, dataset_test_accuracy = model.evaluate(dataset_test)

print('Dataset Test Loss:     %s' % dataset_test_loss)
print('Dataset Test Accuracy: %s' % dataset_test_accuracy)

def plot_dataset_predictions(dataset):

    features, labels = dataset_test.as_numpy_iterator().next()

    predictions = model.predict_on_batch(features).flatten()
    predictions = tf.where(predictions < 0.5, 0, 1)

    print('Labels:      %s' % labels)
    print('Predictions: %s' % predictions.numpy())

    plt.gcf().clear()
    plt.figure(figsize = (15, 15))

    for i in range(9):

        plt.subplot(3, 3, i + 1)
        plt.axis('off')

        plt.imshow(features[i].astype('uint8'))
        plt.title(class_name[predictions[i]])

plot_dataset_predictions(dataset_test)
# ### SAVE & LOAD

model.save('model.keras')

model = tf.keras.models.load_model('model.keras')
# ### PREDICTIONS

def predict(image_file):

    image = tf.keras.preprocessing.image.load_img(image_file, target_size = image_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, 0)

    prediction = model.predict(image)[0][0]

    print('Prediction: {0} | {1}'.format(prediction, ('cracked' if prediction < 0.5 else 'normal')))

def predict_url(image_fname, image_origin):

    image_file = tf.keras.utils.get_file(image_fname, origin = image_origin)
    return predict(image_file)

predict("closeup-shot-black-wheel-tire-texture.jpg")



