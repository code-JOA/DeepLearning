
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense



# Initialising the CNN by creating a class to initialize my CNN
classifier = Sequential()


# Step 1 - CNN - Convolution2D(features , rows , columns)
# adding  the different layers
# I used input_shape of 64,64,3 for TensorFlow backend , 3 reps Red , Blue , Greeen
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - MaxPooling - to reduce the size of our features nodes and reduce time complexity
# By choosing 2*2 we keep the information without loosing features thus reducing the size of
# the feature map thus reducing the complexity of the model without reducing its performance.
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer to improve the model and increase the accuracy
classifier.add(Conv2D(32, (3, 3), activation='relu'))  #input_shape = (64, 64, 3)
classifier.add(MaxPooling2D(pool_size=(2,2)))


# Step 3 - Flattening - involves taking all our pooled feature maps and putting them into
# a single vector
classifier.add(Flatten())


# Step 4 - Full connection - note we can choose a number that isnt too big nor too small.
# Note for the 2nd activation function i used sigmoid since its a binary outcome
# we use softmax if its a multiclass prediction and its output_dim as 1 since
# my prediction is dog or cat. ie i want one prediction dog or cat.
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN - thus loss function
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the CNN to the images to prevent overfitting
from keras.preprocessing.image import ImageDataGenerator

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_size',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier = classifier.fit_generator(training_set,
                                samples_per_epoch = 8000, # since i have 8000 images in my training_set and 2000 in my testset,
                                nb_epoch = 25, # i set epoch to 25 to reduce training time
                                validation_data = test_set,
                                nb_val_samples = 2000)

classifier.save("model.h5")
print("Saved model to disk")

# - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
model = load_model('model.h5')
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)