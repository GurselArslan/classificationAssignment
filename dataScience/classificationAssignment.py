import pandas as pd
import numpy as np
import itertools
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("diabetes.csv")
print(df.head())

#to see types of columns
print(df.info())

######################### PREPROCESSING ########################################

#labelEncoder used to replace categorical values to numerical values. (1=m)
label_encoder = preprocessing.LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Exercise"] = label_encoder.fit_transform(df["Exercise"])

#to see types of columns after encoding
print(df.info())

#to see null values in dataset
print(df.isnull().sum())

#fill nan values with median value
median = df["CalorieIntake"].median()
df["CalorieIntake"].fillna(median,inplace=True)
#to see values after filling
print(df.isnull().sum())

########################## CLASSIFICATION #######################################

#change columns order to easily seperate
df = df[["Pregnancies","Gender","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CalorieIntake","Exercise","SleepDuration","Outcome"]]
inputs = df[["Pregnancies","Gender","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","CalorieIntake","Exercise","SleepDuration"]]
arr1 = inputs.to_numpy()
output = df[["Outcome"]]
arr2 = output.to_numpy()

#building the model with keras
model = Sequential()
model.add(Dense(64,input_dim=12))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('softmax'))

#compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#training the model
model.fit(arr1, arr2, epochs=50, validation_split = 0.1)

#getting predictions
predictions = model.predict_classes(inputs)

#evaluation scores
print(classification_report(arr2,predictions))

#confusion matrix
cm = confusion_matrix(y_true = arr2 , y_pred = predictions)

#function to visualize the confusion matrix 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ["1" , "0"]
plot_confusion_matrix(cm=cm, classes = cm_plot_labels, title="Confusion Matrix")

