#!/usr/bin/env python3
"""
Script para ejecutar el análisis de clasificación con SelectKBest
Basado en el notebook.ipynb
"""

def load_data():
    import pandas as pd

    dataset = pd.read_csv("../files/input/heart_disease.csv")
    y = dataset.pop("target")
    x = dataset.copy()
    x["thal"] = x["thal"].map(
        lambda x: "normal" if x not in ["fixed", "fixed", "reversible"] else x
    )

    return x, y


def make_train_test_split(x, y):
    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        test_size=0.10,
        random_state=0,
    )
    return x_train, x_test, y_train, y_test


def make_pipeline(estimator):
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    transformer = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(dtype="int"), ["thal"]),
        ],
        remainder="passthrough",
    )

    selectkbest = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(
        steps=[
            ("tranformer", transformer),
            ("selectkbest", selectkbest),
            ("estimator", estimator),
        ],
        verbose=False,
    )

    return pipeline


def make_grid_search(estimator, param_grid, cv=5):
    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
    )

    return grid_search


def save_estimator(estimator):
    import pickle

    with open("estimator.pickle", "wb") as file:
        pickle.dump(estimator, file)


def load_estimator():
    import os
    import pickle

    if not os.path.exists("estimator.pickle"):
        return None
    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def train_estimator(estimator):
    from sklearn.metrics import mean_absolute_error

    data, target = load_data()

    x_train, x_test, y_train, y_test = make_train_test_split(
        x=data,
        y=target,
    )

    estimator.fit(x_train, y_train)

    best_estimator = load_estimator()

    if best_estimator is not None:
        saved_mae = mean_absolute_error(
            y_true=y_test, y_pred=best_estimator.predict(x_test)
        )

        current_mae = mean_absolute_error(
            y_true=y_test, y_pred=estimator.predict(x_test)
        )

        if saved_mae < current_mae:
            estimator = best_estimator

    save_estimator(estimator)


def train_logistic_regression():
    from sklearn.linear_model import LogisticRegression

    pipeline = make_pipeline(
        estimator=LogisticRegression(max_iter=10000, solver="saga"),
    )

    param_grid = {
        "selectkbest__k": range(1, 11),
        "estimator__penalty": ["l1", "l2"],
        "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100],
    }

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
    )

    train_estimator(estimator)


def eval_metrics(
    y_train_true,
    y_test_true,
    y_train_pred,
    y_test_pred,
):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    accuracy_train = round(accuracy_score(y_train_true, y_train_pred), 4)
    accuracy_test = round(accuracy_score(y_test_true, y_test_pred), 4)
    balanced_accuracy_train = round(
        balanced_accuracy_score(y_train_true, y_train_pred), 4
    )
    balanced_accuracy_test = round(balanced_accuracy_score(y_test_true, y_test_pred), 4)

    return (
        accuracy_train,
        accuracy_test,
        balanced_accuracy_train,
        balanced_accuracy_test,
    )


def report(
    estimator,
    accuracy_train,
    accuracy_test,
    balanced_accuracy_train,
    balanced_accuracy_test,
):
    print(estimator, ":", sep="")
    print("-" * 80)
    print(f"Balanced Accuracy: {balanced_accuracy_test} ({balanced_accuracy_train})")
    print(f"         Accuracy: {accuracy_test} ({accuracy_train})")


def check_estimator():
    import pickle
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    data, target = load_data()

    x_train, x_test, y_train_true, y_test_true = make_train_test_split(
        x=data,
        y=target,
    )

    estimator = load_estimator()

    y_train_pred = estimator.predict(x_train)
    y_test_pred = estimator.predict(x_test)

    (
        accuracy_train,
        accuracy_test,
        balanced_accuracy_train,
        balanced_accuracy_test,
    ) = eval_metrics(
        y_train_true,
        y_test_true,
        y_train_pred,
        y_test_pred,
    )

    report(
        estimator.best_estimator_,
        accuracy_train,
        accuracy_test,
        balanced_accuracy_train,
        balanced_accuracy_test,
    )


def train_mlp_classifier():
    from sklearn.neural_network import MLPClassifier

    pipeline = make_pipeline(
        estimator=MLPClassifier(max_iter=10000),
    )

    param_grid = {
        "selectkbest__k": range(1, 11),
        "estimator__hidden_layer_sizes": [(h,) for h in range(1, 11)],
        "estimator__learning_rate_init": [0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    estimator = make_grid_search(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
    )

    train_estimator(estimator)


def main():
    """Función principal que ejecuta todo el análisis"""
    print("=== Análisis de Clasificación con SelectKBest ===")
    print()
    
    print("1. Entrenando Regresión Logística...")
    train_logistic_regression()
    print("   ✓ Regresión Logística entrenada")
    
    print("\n2. Evaluando Regresión Logística...")
    check_estimator()
    
    print("\n3. Entrenando MLP Classifier...")
    train_mlp_classifier()
    print("   ✓ MLP Classifier entrenado")
    
    print("\n4. Evaluando MLP Classifier...")
    check_estimator()
    
    print("\n=== Análisis completado ===")


if __name__ == "__main__":
    main()
