# A python script that replicates most of the functionality of the
# associated notebook. Some reporting functions are not included.

# Packages

# Basic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lazypredict
import sklearn

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, f1_score

# Balancing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import imblearn

# Read the X and y values of the data and display a description
def get_data():
    df = pd.read_csv("creditcard.csv")
    desc = df.describe()
    return df,desc
df,desc = get_data()

# Normalize the V columns
for i in range(1,29):
    colname = "V"+str(i)
    df[colname] = df[colname] /desc[colname]["std"]
    
# Apply a log transformation to the 'Amount' column and normalize it
df["Amount"] = np.log(0.01+df["Amount"])
df["Amount"] = (df["Amount"]-df["Amount"].mean()) / df["Amount"].std()

# Normalize time column
df["Time"] = (df["Time"]-df["Time"].mean()) / df["Time"].std()


# Prepare several datasets for training.
X = df.loc[:, df.columns != "Class"]
y = df["Class"]

# Strategy for sampling
# We want the test set to look like the original data set, so we do the train test split first.
# When we rebalance so we can train on balanced data sets.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=123, stratify=y)

print("Total value counts\n",y_train.value_counts())

# Build a dataset for use with the random forest 

# Oversampling with SMOTE
oversample = SMOTE(sampling_strategy = 0.1)
oversample.fit(X_train,y_train)
X_rf,y_rf = oversample.fit_resample(X_train,y_train)
# Undersampling
undersample = RandomUnderSampler(sampling_strategy='majority')
undersample.fit(X_rf,y_rf)
X_rf,y_rf = undersample.fit_resample(X_rf,y_rf)
print("Dataset for random forest classifier\n",y_rf.value_counts())

# Smaller data set for lazy classifier
undersample = RandomUnderSampler(sampling_strategy='majority')
undersample.fit(X_train,y_train)
X_lazy,y_lazy = undersample.fit_resample(X_train,y_train)
print("Dataset for lazy classifier\n",y_lazy.value_counts())


########################################
# Some classifiers
########################################

def random_forest():
    def random_forest():
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_rf,y_rf)
        return clf
    clf = random_forest()

    # Accuracy
    y_train_pred = clf.predict(X_rf)
    print("Training accuracy: ", accuracy_score(y_rf, y_train_pred))
    y_test_pred = clf.predict(X_test)
    print("Test accuracy: ", accuracy_score(y_test, y_test_pred))

    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    # Interpretation: value on the first row, second column is the number of examples that are not fraud
    # but classified as fraud.
    # The second row, first column is the number that are fraud but classified as not fraud.

    # Visualize the confusion matrix.
    plt.rcParams["figure.figsize"]=(6,4)
    sns.heatmap(cm)
random_forest()

############################# SVC

# SVC model
def svc():
    from sklearn.svm import SVC

    # Out of the box, the SVC model is worthless, despite a 99.8% accuracy rate, because it
    # simply predicts all transactions to be nonfraudulent.

    svc_clf = SVC(kernel='linear')
    svc_clf.fit(X_rf,y_rf)
    y_pred = svc_clf.predict(X_test)
    print(accuracy_score(y_pred, y_test.values))
    print("F1: ",f1_score(y_pred, y_test.values))

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
svc()

###################################
# Calibrated Classifier

def calibrated_classifier():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV
    base_clf = GaussianNB()
    base_clf.fit(X_train, y_train)

    calibrated_clf = CalibratedClassifierCV(base_clf,cv="prefit")
    calibrated_clf.fit(X_train,y_train)

    y_pred = calibrated_clf.predict_proba(X_test)[:,1]
    y_pred_adj = [0 if y_pred[i]<0.05 else 1 for i in range(len(y_pred))]

    print("Accuracy: ",accuracy_score(y_pred_adj, y_test.values))
    print("F1: ",f1_score(y_test, y_pred_adj))
    
    cm = confusion_matrix(y_test, y_pred_adj)
    print(cm)
    print(precision_recall_fscore_support(y_test,y_pred_adj))
calibrated_classifier()

##################################
# Lazy classifier
def lazy():
    from lazypredict.Supervised import LazyClassifier
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    model, predictions = clf.fit(X_lazy, X_test, y_lazy, y_test)
    print(model)
lazy()