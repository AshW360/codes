import matplotlib.pyplot as plt
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from pandas import  read_csv
from google.colab import drive
from sklearn.metrics import accuracy_score, classification_report
from itertools import cycle
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import xgboost
from itertools import cycle
drive.mount('/content/drive')
data_set = open('/content/drive/My Drive/Biomarkers_CESC/nblret.csv','r')
from sklearn.preprocessing import LabelEncoder
from collections import Counter
# Oversample and plot imbalanced dataset with SMOTE
from imblearn.over_sampling import SMOTE
# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None, delimiter=',', quotechar='"')
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:,-1]
    return X, y
# load the dataset
X, y = load_dataset(data_set)
# summarize class distribution
counter = Counter(y)
print(counter)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
from sklearn.model_selection import train_test_split
# set aside 20% of train and test data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, shuffle = True, random_state = 8)
# Use the same function above for the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.25, random_state= 8) # 0.8 x 0. = 0.2
# Create the models
# SVM
#svm = LinearSVC(random_state=0) 
#  XGBOOST
xgmodel = xgboost.XGBClassifier(colsample_bytree=0.7,learning_rate=1,max_depth=10,min_child_weight=5)
# Gaussian Naive Bayes
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
# Training the Naive Bayes model on the Training set
#from sklearn.naive_bayes import GaussianNB
#nb = GaussianNB(var_smoothing= 0.000000000001)
#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(C=0.99, multi_class='multinomial', solver='lbfgs')
#ExtraTrees
#from sklearn.ensemble import ExtraTreesClassifier
# Building the model
#etmodel = ExtraTreesClassifier(max_features='log2',n_estimators=100, n_jobs=4, min_samples_split=25,min_samples_leaf=35)
#Random Forest
# importing random forest classifier from assemble module
#from sklearn.ensemble import RandomForestClassifier
#from sklearn import model_selection
# creating a RF classifier
#rf = RandomForestClassifier(random_state= 10,bootstrap = True,max_depth = 90, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 5, n_estimators = 1600) 

# Make it an OvR classifier
#ovr_classifier = OneVsRestClassifier(svm)
ovr_classifier = OneVsRestClassifier(xgmodel)
#ovr_classifier = OneVsRestClassifier(nb)
#ovr_classifier = OneVsRestClassifier(etmodel)
#ovr_classifier = OneVsRestClassifier(rf)
#ovr_classifier = OneVsRestClassifier(lr)


# Fit the data to the OvR classifier
ovr_classifier = ovr_classifier.fit(X_train, y_train)

# Evaluate by means of a confusion matrix
matrix = plot_confusion_matrix(ovr_classifier, X_test, y_test,
                                 cmap='viridis',
                                 normalize='true')
plt.title('Confusion matrix for OvR classifier - xgboost')
plt.show(matrix)
plt.show()


# make predictions for test data
y_pred = ovr_classifier.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Test set Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from sklearn.model_selection import LeaveOneOut
# create loocv procedure
cv = LeaveOneOut()
# evaluate model
scores = cross_val_score(ovr_classifier, X_val, y_val, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Validation Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# Evaluating the model
print("Classification Report \n\n{" + classification_report(
    y_test, y_pred)+ "}")

from sklearn import metrics
metrics.matthews_corrcoef(y_val, y_pred,sample_weight=None)
