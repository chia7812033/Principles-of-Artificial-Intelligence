import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():
    def __init__(self):
        self.x, self.y = self.make_dataset()
        self.X_train, self.y_train, self.X_test, self.y_test = self.train_test_split(self.x, self.y)
        self.losses = []
        self.weights_list = []
        self.X_train = self.normalize(self.X_train)
        self.X_test = self.normalize(self.X_test)

    def make_dataset(self):
        homework_data = pd.read_csv('Problem 2/Averaged homework scores.csv')
        final_data = pd.read_csv('Problem 2/Final exam scores.csv')
        x = pd.concat([homework_data, final_data], axis='columns').to_numpy()
        
        y = pd.read_csv('Problem 2/Results.csv').to_numpy()

        return x, y

    def train_test_split(self, X, y):
        n = 400
        index = np.random.choice(X.shape[0], n, replace=False)

        X_train = np.array(X[index])
        y_train = np.array(y[index]).reshape(-1)
        X_test = np.delete(X, index, axis=0)
        y_test = np.delete(y, index).reshape(-1)

        return X_train, y_train, X_test, y_test

    def fit(self, batch_size, iterations, lr):
        self.losses = []
        self.weights_list = []
        self.weights = np.zeros(self.X_train.shape[1])

        for i in range(iterations):
            #print(f"iter {i}", end=' ')
            dw = self.sgd(batch_size)
            self.update_model_parameters(dw, lr)
            loss = self.compute_loss(self.y_train, self.sigmoid(np.matmul(self.X_train, self.weights)))
            #print(f"loss: {loss}")
            self.losses.append(loss)

        self.update_output_parameters()

    
    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_0 = (1 - y) * np.log(1 - y_pred + 1e-7)
        term_1 = y * np.log(y_pred + 1e-7)
        loss = -np.mean(term_0 + term_1)
        return loss

    def sgd(self, size):
        index = np.random.choice(self.X_train.shape[0], size, replace=False)
        x = self.X_train[index]
        y = self.y_train[index]
        y_pred = self.sigmoid(np.matmul(x, self.weights))
        g = self.compute_gradients(x, y, y_pred)
        return g

    def compute_gradients(self, x, y, y_pred):
        gradients_w = np.matmul(x.T, (y_pred - y)) / y.shape[0]
        return gradients_w
    
    def update_model_parameters(self, dw, lr):
        self.weights = self.weights - lr * dw
        self.weights_list.append(self.weights)

    def update_output_parameters(self):
        self.weights = np.mean(self.weights_list, axis=0)

    def predict(self, X):
        preds = []
        preds = self.sigmoid(np.matmul(X, self.weights))
        pred = [1 if i > 0.5 else 0 for i in preds]
        pred = np.array(pred)
        return pred

    def plot_decision_boundary(self):
        
        x1 = [min(self.X_train[:,0]), max(self.X_train[:,0])]
        x1 = np.array(x1)
        m = -self.weights[0] / self.weights[1]
        x2 = m * x1
        
        fig = plt.figure(figsize=(10,8))
        plt.plot(self.X_train[:, 0][self.y_train==0], self.X_train[:, 1][self.y_train==0], "r*")
        plt.plot(self.X_train[:, 0][self.y_train==1], self.X_train[:, 1][self.y_train==1], "b*")
        plt.xlabel("Averaged homeword scores")
        plt.ylabel("Final exam scores")
        plt.plot(x1, x2, 'k')
        plt.legend(['Adimitted', 'Rejected', 'Decision boundary'])
        plt.grid()
        plt.show()

    def plot_loss(self):
        fig = plt.figure()
        plt.plot(np.arange(0, 1000, 1), self.losses)
        plt.show()

    def test(self, batch_size, iter, lr):
        self.fit(batch_size, iter, lr)
        pred = self.predict(self.X_test)
        loss = self.compute_loss(pred, self.y_test)
        return loss

    def normalize(self, X):
        n = X.shape[1]
        for i in range(n):
            X = (X - X.mean(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            
        return X

model = LogisticRegression()
print(model.test(250, 1000, 0.75))
model.plot_decision_boundary()
