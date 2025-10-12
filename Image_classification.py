
### for simple image classifications using scikit-learn library: - -
import cv2
import os
import pickle #for saving the file
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

##prepare data
input_dir = 'img_classify_parking_lot'
categories = ['empty', 'not_empty']

data = []
labels = [] 

for category_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        #print(img.shape)
        if img is not None: 
            img = resize(img, (15, 15, 3), anti_aliasing=True)  # anti_aliasing : Smooths the image during resizing to avoid aliasing artifacts (jagged edges or pixelation)
            #img = img.resize(100,100)
            data.append(img.flatten()) #flatted is used to make the img a long array(flattening) from being a matrix of 2D and 3channels
            labels.append(category_index)
        else:
            print(f"Warning: Failed to load {img_path}")

data = np.asarray(data)
labels = np.asarray(labels)

print(f"Total samples: {len(data)}, Total labels: {len(labels)}")
#print(len(data))
#print(len(labels)) # they're the same

##train or test split
# we're gonna split the data into traing set and testing set here

x_train, y_train, x_test, y_test = train_test_split(data, labels, test_size= 0.2, random_state=42 ,shuffle= True, stratify= labels)
#print(x_train.shape)
#print(y_train.shape)
#print(x_train.dtype)  # Should be float32 or float64
#print(y_train.dtype)  # Should be int or float
#print(x_train,y_train)
# size for the sets is defoned by train_size =. the data is truely shuffed. Stratify is to take the same proportion of
# each type of  data 
# into the sets as there is in the main directory

##scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# === Step 4: Manual Model Training (Sanity Check) ===
classifier = SVC(C=1, gamma=0.001) # used for classification tasks
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f"Manual fit accuracy: {score * 100:.2f}%")

##train classifier
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100]}] #there are going to be 3*3=9 classifiers

grid_search = GridSearchCV(SVC(), parameters, error_score='raise') #Automatically searches over the grid of hyperparameter combinations.
                                                    #Performs cross-validation for each combination.
                                                    #Selects the best combination based on validation performance.
#grid_search.fit(x_train, y_train) #Trains the model on x_train and y_train using every combination of C and gamma.
                                  #Evaluates each using cross-validation.
grid_search.fit(x_train,y_train)

##test performance
best_estimator = grid_search.best_estimator_  # finding the best classifier of all
y_prediction = best_estimator.predict(x_test)
score = accuracy_score(y_prediction, y_test)

print('{}% of samples are correctly classifies'.format(str(score*100)))

pickle.dump(best_estimator,open('./model.p', 'wb'))