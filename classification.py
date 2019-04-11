import time
from sklearn.metrics import classification_report
import mnist_reader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
print("training data set", np.shape(X_train))
print("testing data set", np.shape(X_test))


def analyzeResult(y_test, y_pred, time1):
    errorCount = 0  # 错误数
    for i in range(10000):
        if y_pred[i] != y_test[i]:
            errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)  # 输出测试错误样本数
    print("the total correct rate is: %.5f" % (1 - (errorCount / float(10000))))  # 输出错误率
    t2 = time.time()
    print("Cost time: %.2fmin, %.4fs." % ((t2 - time1) // 60, (t2 - time1) % 60))  # 测试耗时

    report = classification_report(y_test, y_pred)
    print(report)


# random forest
print("\nRandom Forest:")
t1 = time.time()
classifier = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=60)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
analyzeResult(y_test, y_pred, t1)

# SVM
print("SVM:")
t1 = time.time()
clf = svm.LinearSVC(max_iter=500, C=1.3, tol=1e-4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
analyzeResult(y_test, y_pred, t1)

# adaptive boost
print("Adaptive boosting")
t1 = time.time()
dt = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_samples_leaf=10)
bdt = AdaBoostClassifier(base_estimator=dt, algorithm="SAMME.R", n_estimators=10, learning_rate=1)
bdt.fit(X_train, y_train)
y_pred = bdt.predict(X_test)
analyzeResult(y_test, y_pred, t1)
