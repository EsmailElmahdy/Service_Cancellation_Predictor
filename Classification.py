import saveLoadData
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
import numpy as np

class Classification:
    saveLoad = saveLoadData.SaveLoadData()
    # Constructor
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y

    def Train(self):
        c = 0.0000000000000001

        rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_split=2, min_samples_leaf=2)
        rf.fit(self.X, np.ravel(self.Y))

        knn = KNeighborsClassifier(n_neighbors=1000)
        knn.fit(self.X, np.ravel(self.Y))

        gb = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, ccp_alpha=0.1)
        gb.fit(self.X, np.ravel(self.Y))

        dt = DecisionTreeClassifier(max_depth=2, min_samples_split=2, min_samples_leaf=1, ccp_alpha=0.1)
        dt.fit(self.X, np.ravel(self.Y))

        logreg = LogisticRegression(C=1.0)
        logreg.fit(self.X, np.ravel(self.Y))

        svm = SVC(C=0.1, kernel='rbf')
        svm.fit(self.X, np.ravel(self.Y))

        lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=2, learning_rate=0.1)
        lgbm.fit(self.X, np.ravel(self.Y))

        nb = GaussianNB()
        nb.fit(self.X, np.ravel(self.Y))

        ada_boost = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
        ada_boost.fit(self.X, np.ravel(self.Y))

        self.saveLoad.saveModel(rf, 'RandomForestClassifier')
        self.saveLoad.saveModel(knn, 'KNeighborsClassifier')
        self.saveLoad.saveModel(gb, 'GradientBoostingClassifier')
        self.saveLoad.saveModel(dt, 'DecisionTreeClassifier')
        self.saveLoad.saveModel(logreg, 'LogisticRegression')
        self.saveLoad.saveModel(svm, 'SupportVectorMachineClassifier')
        self.saveLoad.saveModel(nb, 'NaiveBayesClassifier')
        self.saveLoad.saveModel(ada_boost, 'AdaBoostClassifier')
        self.saveLoad.saveModel(lgbm, 'LightGBMClassifier')
        return rf.predict(self.X), knn.predict(self.X), gb.predict(self.X), dt.predict(self.X), logreg.predict(self.X), svm.predict(self.X), nb.predict(self.X), ada_boost.predict(self.X), lgbm.predict(self.X)

    def Test(self):
        rf = self.saveLoad.loadModel('RandomForestClassifier')
        knn = self.saveLoad.loadModel('KNeighborsClassifier')
        gb = self.saveLoad.loadModel('GradientBoostingClassifier')
        dt = self.saveLoad.loadModel('DecisionTreeClassifier')
        logreg = self.saveLoad.loadModel('LogisticRegression')
        svm = self.saveLoad.loadModel('SupportVectorMachineClassifier')
        nb = self.saveLoad.loadModel('NaiveBayesClassifier')
        ada_boost = self.saveLoad.loadModel('AdaBoostClassifier')
        lgbm = self.saveLoad.loadModel('LightGBMClassifier')
        rf_pr = rf.predict(self.X)
        ##############
        knn_pr = knn.predict(self.X)
        ##############
        gb_pr = gb.predict(self.X)
        ##############
        dt_pr = dt.predict(self.X)
        ##############
        lr_pr = logreg.predict(self.X)
        ##############
        sv_pr = svm.predict(self.X)
        ##############
        nb_pr = nb.predict(self.X)
        ##############
        ada_boost_pr = ada_boost.predict(self.X)
        ##############
        lgbm_pr = lgbm.predict(self.X)
        return rf_pr, knn_pr, gb_pr, dt_pr, lr_pr, sv_pr, nb_pr, ada_boost_pr, lgbm_pr