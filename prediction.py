import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np

import os

import random

from sklearn.preprocessing import StandardScaler

model = tf.keras.models.load_model("captcha_reader_simple_nn.h5")
model_comparison = tf.keras.models.load_model("captcha_reader_comparison_CNN_solution.h5")

labels = {0: '2', 4: '6', 13: 'm', 9: 'd', 3: '5', 14: 'n', 1: '3', 12: 'g', 6: '8', 2: '4', 10: 'e', 18: 'y', 11: 'f', 16: 'w', 15: 'p', 5: '7', 8: 'c', 17: 'x', 7: 'b'}
# test=""
# for i in range(len(labels)):
#     test+="'"+labels[i]+"'"
#     if i != len(labels)-1:
#         test+=", "
# print("\nAvailable characters:",test,"\n")
def predict (captcha_path):
    captcha_value = captcha_path.replace('./data/samples/', '').replace('.png', '')

    img = mpimg.imread(captcha_path)
    plt.imshow(img)
    plt.show()
    x = []
    for i in range(5):
            x.append(img[:,30+i*21:50+i*21,:])
    x = np.array(x)
    x = x.reshape(-1,4000)
    x = x.astype(float)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = model.predict(x)
    y = np.argmax(y, axis = 1) # Returns the most likely

    prediction = ""
    for i in range(len(y)) :
        prediction += labels[y[i]]
    print("Predicted value: ",prediction)
        
    print("Captcha value: ", captcha_value)

def predict_with_CNN (captcha_path):
    captcha_value = captcha_path.replace('./data/samples/', '').replace('.png', '')

    img = mpimg.imread(captcha_path)
    plt.imshow(img)
    plt.show()
    x = []
    for i in range(5):
            x.append(img[:,30+i*21:50+i*21,:])
    x = np.array(x)
    # x = x.reshape(-1,4000)
    x = x.astype(float)
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)
    y = model_comparison.predict(x)
    y = np.argmax(y, axis = 1) # Returns the most likely

    prediction = ""
    for i in range(len(y)) :
        prediction += labels[y[i]]
    print("Predicted value: ",prediction)
        
    print("Captcha value: ", captcha_value)
        
class IncorrectCharacter:
    def __init__(self, predictedValue, actualValue):
        self.predictedValue = predictedValue
        self.actualValue = actualValue

def characters_errors (files):
    incorrectCharacters = []
    for file in files:
        img = mpimg.imread('./data/samples/'+file)
        x = []
        for i in range(5):
            x.append(img[:,30+i*21:50+i*21,:])
        x = np.array(x)
        # x = x.reshape(-1,4000)
        x = x.astype(float)
        # scaler = StandardScaler()
        # x = scaler.fit_transform(x)
        y = model_comparison.predict(x)
        y = np.argmax(y, axis = 1) # Returns the most likely
        for i in range(len(y)) :
            predictedValue = labels[y[i]]
            actualValue = file[i]
            if predictedValue != actualValue:
                print("Predicted Character: ",predictedValue, "\tActual Character: ", actualValue)
                incorrectCharacters.append(IncorrectCharacter(predictedValue,actualValue))
    return incorrectCharacters


files = []
for file in os.listdir('data/samples'):
    if file.endswith(".png"):
        files.append(file)

incorrectCharacters=characters_errors(files)
print("Total characters in the dataset: ", len(files)*5)
print("Characters predicted incorrectly out of the whole dataset: ",len(incorrectCharacters))
print("Ratio: ", len(incorrectCharacters)/(len(files)*5)*100,"%")

occurrences = {}
for incorrectCharacter in incorrectCharacters:
    if incorrectCharacter.actualValue not in occurrences:
        occurrences[incorrectCharacter.actualValue]=0
    else:
        occurrences[incorrectCharacter.actualValue]+=1
occurrences = {key: val for key, val in sorted(occurrences.items(), key = lambda ele: ele[1], reverse = True)} # Sorting by number of occurrences descending
print(occurrences)    

occurrencesRelative = {key: value / len(incorrectCharacters) for key, value in occurrences.items()}


fig = plt.figure()
occurrencesPlot=fig.add_subplot(121)
occurrencesPlot.set_title("Occurrences")
plt.bar(range(len(occurrences)), list(occurrences.values()), align='center')
plt.xticks(range(len(occurrences)), list(occurrences.keys()))
occurrencesRelativePlot=fig.add_subplot(122)
occurrencesRelativePlot.set_title("Relative Occurrences")
plt.bar(range(len(occurrencesRelative)), list(occurrencesRelative.values()), align='center')
plt.xticks(range(len(occurrencesRelative)), list(occurrencesRelative.keys()))
plt.show()


for i in range(5):   
    file = files[random.randint(0,1039)]
    print('\nWith the model I created:')
    predict('./data/samples/'+file)
    print('With the model using CNN:')
    predict_with_CNN('./data/samples/'+file)
    print("\n")