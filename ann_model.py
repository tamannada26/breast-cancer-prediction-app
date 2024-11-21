# ann_model.py

from sklearn.neural_network import MLPClassifier

def train_ann_model(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500)
    model.fit(X_train, y_train)
    return model
