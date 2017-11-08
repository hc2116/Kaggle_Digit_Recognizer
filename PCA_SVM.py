

#train=open("OneDrive/Kaggle_Daddle/train.csv").readlineshead
#N=3
#with open("OneDrive/Kaggle_Daddle/train.csv") as myfile:
#    head = [next(myfile) for x in xrange(N)]


import numpy as np

#train = np.genfromtxt('OneDrive/Kaggle_Daddle/train.csv', delimiter=',',skip_header=1,skip_footer=41900)

train = np.genfromtxt('OneDrive/Kaggle_Daddle/train.csv', delimiter=',',skip_header=1)

test = np.genfromtxt('OneDrive/Kaggle_Daddle/test.csv', delimiter=',',skip_header=1)


from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import neighbors


X_train=np.delete(train,0,1)-np.delete(train,0,1).mean(axis=0)
y_train=train[0:train.shape[0],0]
y_train=y_train.astype(int)


X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42)

y_train_train=y_train_train.astype(int)
y_train_test=y_train_test.astype(int)


n_faces = X_train.shape[0]
n_pix = X_train.shape[1]
n_components=100

pca=PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train_train)

X_train_train_pca=pca.transform(X_train_train)
X_train_test_pca=pca.transform(X_train_test)


lin_clf = svm.LinearSVC()
lin_clf.fit(X_train_train_pca,y_train_train)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train_train_pca,y_train_train)

clf_NN = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
clf_NN.fit(X_train_train_pca,y_train_train)


y_train_pred_lin = lin_clf.predict(X_train_test_pca)
y_train_pred = clf.predict(X_train_test_pca)
y_train_pred_NN = clf_NN.predict(X_train_test_pca)

print(classification_report(y_train_test, y_train_pred_lin))
print(classification_report(y_train_test, y_train_pred))
print(classification_report(y_train_test, y_train_pred_NN))

n_components=100
X_test=test
pca=PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)

lin_clf = svm.LinearSVC()
lin_clf.fit(X_train_pca,y_train)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train_pca,y_train)

y_pred_lin = lin_clf.predict(X_test_pca)
y_pred = clf.predict(X_test_pca)

number_labels=np.array(range(len(y_pred)))
full=np.vstack((number_labels,y_pred)).T
full=np.vstack((np.array(["ImageID","Label"]),full.astype(str)))


a.tofile('foo.csv',sep=',',format='%10.5f')
np.savetxt("pred.csv", full, delimiter=",",fmt="%s")





