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
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(100, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(ds_train_xy.batch(32), epochs=30)

model.evaluate(ds_test_xy.batch(32))

#look at performance on train set
for i, (image, label) in enumerate(ds_train_xy.shuffle(1000).take(3)):
    image_resize = tf.expand_dims(image, 0)
    pred = model.predict(image_resize)
    plt.subplot(3, 1, i + 1)
    plt.imshow(image)
    plt.title('Predicted: ' + class_names[np.argmax(pred)] + ', Actual: ' + class_names[label])
    plt.axis('off')
plt.show()

# look as performance on test set
for i, (image, label) in enumerate(ds_test_xy.shuffle(1000).take(3)):
    image_resize = tf.expand_dims(image, 0)
    pred = model.predict(image_resize)
    plt.subplot(3, 1, i + 1)
    plt.imshow(image)
    plt.title('Predicted: ' + class_names[np.argmax(pred)] + ', Actual: ' + class_names[label])
    plt.axis('off')
plt.show()

#perform data augmentation
def augment_image(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_crop(x, size=[32, 32, 3])
    x = tf.image.random_contrast(x, 0.2, 0.5)
    return x, y

ds_train_augmented = ds_train_xy.map(augment_image)
# concatenate original data set with augmented one
ds_train_xy_augmented = ds_train_xy.concatenate(ds_train_augmented)

model_2 = keras.Sequential([
    keras.layers.Conv2D(32, 3, padding='same', input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(32, 3, padding='same'),
    keras.layers.Lambda(tf.nn.local_response_normalization),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same'),
    keras.layers.Lambda(tf.nn.local_response_normalization),
    keras.layers.Activation('relu'),
    keras.layers.Conv2D(64, 3, padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(100, activation='softmax')
])

model_2.summary()

model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_2 = model_2.fit(ds_train_xy_augmented.batch(32), epochs=30)

model_2.evaluate(ds_test_xy.batch(32))

# look as performance on test set
for i, (image, label) in enumerate(ds_test_xy.shuffle(1000).take(3)):
    image_resize = tf.expand_dims(image, 0)
    pred = model_2.predict(image_resize)
    plt.subplot(3, 1, i + 1)
    plt.imshow(image)
    plt.title('Predicted: ' + class_names[np.argmax(pred)] + ', Actual: ' + class_names[label])
    plt.axis('off')
plt.show()

# Implement ResNet-34
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

model_3 = keras.models.Sequential()
model_3.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[32, 32, 3], padding='same', use_bias=False))
model_3.add(keras.layers.BatchNormalization())
model_3.add(keras.layers.Activation('relu'))
model_3.add(keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model_3.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model_3.add(keras.layers.GlobalAveragePooling2D())
model_3.add(keras.layers.Flatten())
model_3.add(keras.layers.Dense(100, activation='softmax'))

model_3.summary()

model_3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_3 = model_3.fit(ds_train_xy_augmented.batch(32), epochs=30)

model_3.evaluate(ds_test_xy.batch(32))
#-------------------------------
#RESNET

def preprocess(image, label):
    #resized_image = tf.image.resize(image, (224, 224))
    final_image = keras.applications.resnet50.preprocess_input(image)
    return final_image, label

ds_train_xy_resnet = ds_train_xy.map(preprocess)
ds_test_xy_resnet = ds_test_xy.map(preprocess)

base_model_4 = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
avg = keras.layers.GlobalAveragePooling2D()(base_model_4.output)
output = keras.layers.Dense(100, activation='softmax')(avg)
model_4 = keras.Model(inputs=base_model_4.input, outputs=output)

for layer in base_model_4.layers:
    layer.trainable = False

model_4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_4 = model_4.fit(ds_train_xy_resnet.batch(32), epochs=30)

model_4.evaluate(ds_test_xy_resnet.batch(32))

# now unfreeze the base model

for layer in base_model_4.layers:
    layer.trainable = True

model_4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_4 = model_4.fit(ds_train_xy_resnet.batch(32), epochs=10)

model_4.evaluate(ds_test_xy_resnet.batch(32))






