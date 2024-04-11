import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import numpy as np


def performance_metrics(y_test, y_test_pred):
    # regression evaluation metrics
    r2 = r2_score(y_test, y_test_pred)
    mae = np.mean(np.abs(y_test - y_test_pred))  # mean absolute error
    mse = np.mean(np.square(y_test - y_test_pred))  # mean squared error

    print("R-squared:", r2)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)

    plt.scatter(y_test, y_test_pred)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices Scatter Plot")
    plt.show()

    # Histogram of the differences in %-s between actual and predicted values
    plt.hist((y_test - y_test_pred) / np.maximum(y_test, y_test_pred), bins=20)
    plt.xlabel("Price Difference")
    plt.ylabel("Frequency")
    plt.title("Difference Between Actual and Predicted Prices")
    plt.axvline(x=0, color="r", linestyle="dashed", linewidth=0.8)  # add a zero line
    plt.show()
