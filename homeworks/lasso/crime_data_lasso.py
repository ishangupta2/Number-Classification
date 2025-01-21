if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    input_vars = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']

    x_train = df_train.drop('ViolentCrimesPerPop', axis=1)
    y_train = df_train['ViolentCrimesPerPop']
    x_test = df_test.drop('ViolentCrimesPerPop', axis=1)
    y_test = df_test['ViolentCrimesPerPop']

    d = x_train.shape[1]

    lambda_max = 2.0 * np.linalg.norm(x_train.T @ (y_train - np.mean(y_train)), np.inf)
    w = np.zeros((d, ))

    lambdas = []
    non_zeros = []
    train_mse = []
    test_mse = []

    reg_paths = {
        'agePct12t29': [], 
        'pctWSocSec': [], 
        'pctUrban': [], 
        'agePct65up': [], 
        'householdsize': []
    }

    curr_lambda = lambda_max
    while (curr_lambda >= 0.01):
        lambdas.append(curr_lambda)

        w, b = train(x_train, y_train, curr_lambda)
        non_zeros.append(np.count_nonzero(w))

        y_train_pred = x_train @ w + b
        y_test_pred = x_test @ w + b

        train_mse.append(np.mean(y_train - y_train_pred) ** 2)
        test_mse.append(np.mean(y_test - y_test_pred) ** 2)

        for input in input_vars:
            reg_paths[input].append(w[x_train.columns.get_loc(input)])

        curr_lambda /= 2


    # Plot C
    plt.figure(figsize = (10, 5))
    plt.title("Non-zero weights vs. Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("# of non-zero weights")
    plt.xscale('log')
    plt.plot(lambdas, non_zeros)
    plt.show()

    # Plot D
    plt.figure(figsize = (10, 5))
    for input in input_vars:
        plt.plot(lambdas, reg_paths[input], label=f'{input}')
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel("Regularization paths")
    plt.legend()
    plt.show()

    # Plot E
    plt.figre(figsize = (10, 5))
    plt.plot(lambdas, train_mse, label="Train_MSE")
    plt.plot(lambdas, test_mse, label="Test_MSE")
    plt.xscale('log')
    plt.xlabel("Lambda")
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

    # F
    _lambda = 30
    w = train(x_train, y_train, _lambda)[0]
    print(x_train_df[np.argmax(w)])
    print(y_train )

if __name__ == "__main__":
    main()
