import os
import matplotlib.pyplot as plt  # Corrected import statement
import tensorflow as tf

# Definindo e quantificando imagens de treino e validação
dataset_dir = os.path.join(os.getcwd(), "data")

dataset_train_dir = os.path.join(dataset_dir, "train")
dataset_train_fresh_len = len(os.listdir(
    os.path.join(dataset_train_dir, "fresh")))
dataset_train_half_fresh_len = len(os.listdir(
    os.path.join(dataset_train_dir, "half-fresh")))
dataset_train_spoiled_len = len(os.listdir(
    os.path.join(dataset_train_dir, "spoiled")))

dataset_validation_dir = os.path.join(dataset_dir, "valid")
dataset_validation_fresh_len = len(os.listdir(
    os.path.join(dataset_validation_dir, "fresh")))
dataset_validation_half_fresh_len = len(os.listdir(
    os.path.join(dataset_validation_dir, "half-fresh")))
dataset_validation_spoiled_len = len(os.listdir(
    os.path.join(dataset_validation_dir, "spoiled")))

print("Train Fresh: %s" % dataset_train_fresh_len)
print("Train Half-Fresh: %s" % dataset_train_half_fresh_len)
print("Train Spoiled: %s" % dataset_train_spoiled_len)

print("Validation Fresh: %s" % dataset_validation_fresh_len)
print("Validation Half-Fresh: %s" % dataset_validation_half_fresh_len)
print("Validation Spoiled: %s" % dataset_validation_spoiled_len)

# Configurações de imagem
image_width = 416
image_height = 416
image_color_channel = 3
iamge_color_channel_size = 255
image_size = (image_width, image_height)
image_shape = image_size + (image_color_channel,)

# Configurações de remessas de treinamento
batch_size = 32
epochs = 20
learning_rate = 0.0001

class_name = ["fresh",  "half-fresh", "spoiled"]

# Configurações de treinamento
dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_train_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

# Configurações de treinamento
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

print("Validation Dataset Cardinality: %d" % tf.data.experimental.cardinality(dataset_validation))
print("Test Dataset Cardinality: %d " %tf.data.experimental.cardinality(dataset_test))

def plot_dataset(dataset):
    plt.clf()  # Corrected clearing the figure
    plt.figure(figsize=(15, 15))
    
    for features, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.axis("off")
            
            plt.imshow(features[i].numpy().astype("uint8"))  # Corrected typo
            plt.title(class_name(labels[i]))  # Corrected typo
            
plot_dataset(dataset_train)