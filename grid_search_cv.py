# grid_search_cv.py

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def grid_search_ann(X_train, y_train):
    # Hyperparameter grid
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100,), (100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'max_iter': [200, 300, 500]
    }
    
    model = MLPClassifier()
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_
