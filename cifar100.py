import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# get the train and test data sets for CIFAR-100
ds_train, ds_info_train = tfds.load('cifar100', split='train', with_info=True)
ds_test, ds_info_test = tfds.load('cifar100', split='test', with_info=True)

# get the size of the traing set
training_size = ds_train.reduce(0, lambda x, _: x+1).numpy()
# get the size of the test set
test_size = ds_test.reduce(0, lambda x, _: x+1).numpy()
print('There are '+ str(training_size) +' images in the training set')
print('There are '+ str(test_size) +' images in the test set')

# get the label names
class_names = ds_info_train.features['label'].names
coarse_class_names = ds_info_train.features['coarse_label'].names

print('There are ' + str(len(class_names)) + ' fine labels')
print('There are ' + str(len(coarse_class_names)) + ' coarse labels')

# get the shape of the image in the train set
print(ds_train.element_spec['image'].shape)

# show images with fine labels
for i, example in enumerate(ds_train.shuffle(1000).take(12)):
    image = example['image']
    label = example['label']
    plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.title(class_names[label])
    plt.axis('off')
plt.show()

# show images with coarse labels
for i, example in enumerate(ds_train.shuffle(1000).take(12)):
    image = example['image']
    coarse_label = example['coarse_label']
    plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.title(coarse_class_names[coarse_label], fontsize = 6)
    plt.axis('off')
plt.show()

def extract_image_and_label(data):
    x = data['image']
    y = data['label']
    return x, y

# get the x and y values for the train and test sets
ds_train_xy = ds_train.map(extract_image_and_label)
# do the same for the test set:
ds_test_xy = ds_test.map(extract_image_and_label)

# Build a CNN to classify images
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(100, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(ds_train_xy.batch(32), epochs=30)

model.evaluate(ds_test_xy.batch(32))





