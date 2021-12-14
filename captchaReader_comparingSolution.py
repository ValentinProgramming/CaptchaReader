import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

import os

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import cv2

firstSampleDisplay=True

class TrainingCharacterSample:
    def __init__(self, x, y):
        self.x = x
        self.y = y

trainingData = []

for file in os.listdir('data/samples'):
    if file.endswith(".png"):
        img = mpimg.imread('data/samples/'+file)
        
        if firstSampleDisplay: # Shows the captchas' splitting used as an example (only for the first captcha)
            fig = plt.figure()
            firstChar=fig.add_subplot(151)
            plt.imshow(img[:,30:50]) # First character
            secondChar=fig.add_subplot(152)
            plt.imshow(img[:,51:71]) # Second character
            thirdChar=fig.add_subplot(153)
            plt.imshow(img[:,72:92]) # Third character
            fourthChar=fig.add_subplot(154)
            plt.imshow(img[:,93:113]) # Fourth character
            fifthChar=fig.add_subplot(155)
            plt.imshow(img[:,114:134]) # Fifth character
            firstChar.set_title('First character')
            secondChar.set_title('Second character')  
            thirdChar.set_title('Third character')
            fourthChar.set_title('Fourth character')  
            fifthChar.set_title('Fifth character')
            plt.show()

        firstSampleDisplay=False

        for i in range(5):
            x = img[:,30+i*21:50+i*21]
            y = file[i]
            trainingData.append(TrainingCharacterSample(x,y))
       
for i in range(1):   
    plt.imshow(trainingData[i].x)
    plt.show()
    print(trainingData[i].y)
    print(trainingData[i].x.shape)

trainingDataInputCharacters = np.array([sample.x for sample in trainingData])
trainingDataOutputCharacters =  np.array([sample.y for sample in trainingData])

print(trainingDataOutputCharacters[1])
outputEncodedLabels = LabelEncoder().fit_transform(trainingDataOutputCharacters)
print(outputEncodedLabels[1])
outputOneHotEncodedLabels = OneHotEncoder(sparse = False).fit_transform(outputEncodedLabels.reshape(len(outputEncodedLabels),1))
print(outputOneHotEncodedLabels[1])

labels = {outputEncodedLabels[i] : trainingDataOutputCharacters[i] for i in range(len(trainingDataOutputCharacters))}
print(labels)

trainingDataInputCharacters = trainingDataInputCharacters.astype(float)


char_train, char_test, output_train, output_test = train_test_split(trainingDataInputCharacters, outputOneHotEncodedLabels, test_size=0.2, random_state=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3), padding='same', input_shape=(50,20,4)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))

categories = outputOneHotEncodedLabels.shape[1]

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1500))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(categories))
model.add(tf.keras.layers.Activation("softmax"))

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam' ,
              metrics = ['accuracy'])

print(model.summary())

batch_size = 150
epochs = 20

history = model.fit(char_train, output_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(char_test, output_test),
          shuffle=True)

loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

loss_validation_curve = history.history["val_loss"]
acc_validation_curve = history.history["val_accuracy"]

fig = plt.figure()
loss=fig.add_subplot(121)
loss.set_title("Loss")
plt.plot(loss_curve, label="Train")
plt.plot(loss_validation_curve, label="Validation")
plt.legend(loc='upper left')
accuracy=fig.add_subplot(122)
accuracy.set_title("Accuracy")
plt.plot(acc_curve, label="Train")
plt.plot(acc_validation_curve, label="Validation")
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.show()

loss, accuracy = model.evaluate(char_test, output_test)
print("Test Loss on Test dataset: ", loss)
print("Test Accuracy on Test dataset: ", accuracy)

model.save('captcha_reader_comparison_CNN_solution.h5')




