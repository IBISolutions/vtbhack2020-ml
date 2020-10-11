import tensorflow as tf
from keras import callbacks
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.utils import to_categorical
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

sn.set(font_scale=1.4)

IMAGE_SIZE = (48, 48)


def load_data_test():
    """
    Загружаем одно тестовое изображение
    :return:
    """
    try:
        images = []
        image_test = cv2.imread('/content/drive/My Drive/VTB Test/test_polo.jpg')
        image_test = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_test = cv2.resize(image, IMAGE_SIZE)
        images.append(image_test)
        images = np.array(images, dtype='float32')

        return images
    except Exception as error:
        print("cv2 image:", error)


def load_data():
    """
      Load the data:
    """

    class_names = ['Hyundai Solaris sedan', 'KIA Rio sedan', 'SKODA OCTAVIA sedan', 'Volkswagen Polo sedan',
                   'Volkswagen Tiguan']

    class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
    print("class_names_label: ", class_names_label)

    nb_classes = len(class_names)
    print("nb_classes: ", nb_classes)

    datasets = ['/content/drive/My Drive/VTB']
    output = []

    # for dataset in datasets:
    if datasets:
        dataset = datasets[0]

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):

            print("folder: ", folder)
            if folder == ".ipynb_checkpoints":
                print("Пропускаем: ", folder)
                continue

            check_folder = ['Hyundai Solaris sedan', 'KIA Rio sedan', 'SKODA OCTAVIA sedan', 'Volkswagen Polo sedan',
                            'Volkswagen Tiguan']
            if not folder in check_folder:
                print("Пропускаем: ", folder)
                continue

            label = class_names_label[folder]
            print("label: ", label)

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):

                if file == ".DS_Store":
                    print("Пропуск: ", file)
                    continue

                if not ".jpeg" in file and not ".png" in file and not ".jpg" in file:
                    print("Not .jpeg: ", file)
                    continue

                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                try:
                    # Open and resize the img
                    img_width, img_height = 48, 48
                    img = image.load_img(img_path, target_size=(img_width, img_height))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                except Exception as error:
                    print("img:", img_path, error)
                    continue

                # Append the image and its corresponding label to the output
                images.append(img)
                labels.append(label)

        images = np.array(images, dtype='float32')

        labels = np.array(labels, dtype='int32')

    return (images, labels)


input_shape = (48, 48, 3)

# note #
# datagen = ImageDataGenerator(rotation_range=90)
# (X_train, Y_train) = load_data()
# it = datagen.flow_from_directory(X_train, Y_train)
# it = datagen.flow_from_directory(X_train, Y_train, labels="inferred", label_mode="int")
# end note #

(X_train, Y_train) = load_data()

# Y_train = to_categorical(y_train, 4)

# resize train set
X_train_resized = []
for img in X_train:
    X_train_resized.append(np.resize(img, input_shape) / 255)

X_train_resized = np.array(X_train_resized)
print(X_train_resized.shape)

# We build the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# We freeze every layer in our base model so that they do not train, we want that our feature extractor stays as before --> transfer learning
for layer in base_model.layers:
    layer.trainable = False
    print('Layer ' + layer.name + ' frozen.')

# We take the last layer of our the model and add it to our classifier
last = base_model.layers[-1].output
x = Flatten()(last)
x = Dense(1000, activation='relu', name='fc1')(x)
x = Dropout(0.3)(x)
x = Dense(5, activation='softmax', name='predictions')(x)

model = Model(base_model.input, x)

# We compile the model
# new new
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# note #
# model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# end note #

# We start the training
epochs = 10
batch_size = 128
# We train it
# note #
# model.fit_generator(it, steps_per_epoch=313, batch_size=batch_size)
# end note #

model.fit(X_train_resized, Y_train,
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.0)

# Схраняем модель
model.save('/content/drive/My Drive/VTB_Models/new_model.h5')

test_images = load_data_test()
predictions = model.predict(test_images)

print("predictions: ", predictions)
