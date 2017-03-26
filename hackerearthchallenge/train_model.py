from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.optimizers import RMSprop
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
import numpy as np
import h5py
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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
        with open("/home/suraj/Repositories/Hacker Earth Challenge/datasets/x_dataset.pkl", "rb") as dataset:
            self.x = pickle.load(dataset)
        with open("/home/suraj/Repositories/Hacker Earth Challenge/datasets/y_dataset.pkl", "rb") as dataset:
            self.y = pickle.load(dataset)

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print self.model.best_params_
        print "Training Accuracy: {0}".format(self.model.score(self.X_train, self.y_train))
        print "Validation Accuracy: {0}".format(self.model.score(self.X_test, self.y_test))
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=3)
        print " Training input shape : {0}, Target shape: {1}".format(np.shape(self.X_train), np.shape(self.y_train))
        print "Validation input shape : {0}, Target shape: {1}".format(np.shape(self.X_test), np.shape(self.y_test))

    def randomforest_initialize(self, estimators, depth, leaf_nodes, features):
        """
        Tuning parameters for Random Forests
        
        
        :param estimators: This is the number of trees you want to build before taking the maximum voting or averages of predictions
        :param depth: Max depth of each tree
        :param leaf_nodes: A smaller leaf makes the model more prone to capturing noise in train data
        :param features: These are the maximum number of features Random Forest is allowed to try in individual tree
        :return: 
        """
        forest = RandomForestClassifier(verbose=True, n_jobs= -1, random_state=3)
        self.model = GridSearchCV(forest, param_grid= {'n_estimators': estimators, 'max_depth': depth, 'max_features': features, 'min_samples_leaf': leaf_nodes }, verbose=True)

    def adaboost_initialize(self, estimator_range, learning_rate):
        adaboost = AdaBoostClassifier()
        self.model = GridSearchCV(adaboost, param_grid= {'n_estimators': estimator_range, 'learning_rate': learning_rate}, scoring=self.auc_scorer, verbose=True, n_jobs=-1)

    def gradient_boost_initialize(self, estimators, lr, depth):
        gdboost = GradientBoostingClassifier(verbose=True)
        self.model = GridSearchCV(gdboost, param_grid={'n_estimators': estimators, 'learning_rate': lr , 'max_depth': depth}, scoring= self.auc_scorer, verbose=True, n_jobs=-1)

    def neural_network_initialize(self, hidden_layer, optimizer, regularization, minit_batch, lr, iterations, lr_init):
        net = MLPClassifier(verbose=True, random_state= 3)
        self.model = GridSearchCV(net, param_grid={'hidden_layer_sizes': hidden_layer,
                                                   'solver': optimizer, 'alpha': regularization,
                                                   'batch_size': minit_batch, 'learning_rate': lr,
                                                   'max_iter': iterations, 'learning_rate_init': lr_init
        }, verbose=True)



def run():
    model = Model()
    model.load_dataset()
    model.split_dataset()
    model.randomforest_initialize(estimators=[1000], depth=[200], leaf_nodes=[4], features=[None])
    model.train_model()

if __name__ == "__main__":
    run()