from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_learning_curves(model, X, y):

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(150, len(X_train), 150):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(f1_score(y_train_predict, y_train[:m]))
        val_errors.append(f1_score(y_val_predict, y_val))

    plt.plot(train_errors, "r-+", lineWidth=2, label = "train")
    plt.plot(val_errors, "b-", lineWidth=3, label = "val")
    plt.show()
    return
