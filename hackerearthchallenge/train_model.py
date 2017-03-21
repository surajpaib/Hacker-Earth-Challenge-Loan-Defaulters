from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split


class Model(object):
    def __init__(self):
        self.x = []
        self.y = []
        self.model = None
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)

    def load_dataset(self):
        with open("/home/suraj/Repositories/Hacker Earth Challenge/x_dataset.pkl", "rb") as dataset:
            self.x = pickle.load(dataset)
        with open("/home/suraj/Repositories/Hacker Earth Challenge/y_dataset.pkl", "rb") as dataset:
            self.y = pickle.load(dataset)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print "Training Accuracy: {0}".format(self.model.score(self.X_train, self.y_train))
        print "Validation Accuracy: {0}".format(self.model.score(self.X_test, self.y_test))
        print "AUC ROC SCORE {0} ".format(roc_auc_score(self.y_test, self.model.predict(self.X_test)))
        joblib.dump(self.model, 'rf_model.pkl')


    def reshape_tensorflow(self):
        self.x = np.reshape(self.x, (np.shape(self.x)[0], 1, np.shape(self.x)[1]))
        self.y = np.reshape(self.y, (np.shape(self.y)[0], 1))

    def keras_train_model(self):
        print "Y = {0}".format(np.shape(self.y))
        print "X = {0}".format(np.shape(self.x))
        model = Sequential()
        model.add(Dense(15, input_shape=(1, np.shape(self.x)[2])))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Activation('sigmoid'))

        model.summary()

        model.compile(optimizer='adam', loss='msle', metrics=['accuracy'])
        model.fit(self.x, self.y,batch_size=1000, nb_epoch=500, validation_split= 0.30)
        model.save('model.h5')

    def split_dataset(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.3, random_state=3)
        print " Training input shape : {0}, Target shape: {1}".format(np.shape(self.X_train), np.shape(self.y_train))
        print "Validation input shape : {0}, Target shape: {1}".format(np.shape(self.X_test), np.shape(self.y_test))

    def randomforest_initialize(self, estimators):
        forest = RandomForestClassifier(verbose=True, n_jobs= -1)
        self.model = GridSearchCV(forest, param_grid= {'n_estimators': estimators}, scoring=self.auc_scorer, verbose=True)

    def adaboost_initialize(self, estimator_range, learning_rate):
        adaboost = AdaBoostClassifier()
        self.model = GridSearchCV(adaboost, param_grid= {'n_estimators': estimator_range, 'learning_rate': learning_rate}, scoring=self.auc_scorer, verbose=True, n_jobs=-1)

    def gradient_boost_initialize(self, estimators, lr, depth):
        gdboost = GradientBoostingClassifier(verbose=True)
        self.model = GridSearchCV(gdboost, param_grid={'n_estimators': estimators, 'learning_rate': lr , 'max_depth': depth}, scoring= self.auc_scorer, verbose=True, n_jobs=-1)


def run():
    model = Model()
    model.load_dataset()
    model.split_dataset()
    model.gradient_boost_initialize(estimators=[100], lr=[1], depth= [7])
    model.train_model()

if __name__ == "__main__":
    run()