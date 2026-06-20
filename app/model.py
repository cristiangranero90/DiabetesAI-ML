from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from app.data_layer import DataLayer

dl = DataLayer()


class SimpleSVCModel:
    def __init__(self):
        self.model = SVC(probability=True)
        self.is_trained = False
        self.y_test = []
        self.x_test = []

        #self.train()

    def trained(self):
        return self.is_trained

    def prepare_data(self):
        data_df = dl.get_training_data_sp()
        lista_caract = [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]
        lista_etiq = ['Outcome']
        y = data_df[lista_etiq]
        X = data_df[lista_caract]
        return X, y
    

    def train(self):
        val_x, val_y = self.prepare_data()
        X_train, self.x_test, y_train , self.y_test = train_test_split(val_x, val_y, test_size=0.5, random_state=42)       
        self.model.fit(X_train.values, y_train.values.ravel())
        self.is_trained = True

    def predict_fun(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet")
        else:
            y_test_pred = self.model.predict(self.x_test.values)
            y_pred = self.model.predict(X)
            return accuracy_score(self.y_test, y_test_pred), y_pred.tolist()
        

    def predict_fun_proba(self, X):
        if not self.is_trained:
            raise Exception("Model is not trained yet")
        return self.model.predict_fun_proba(X)
