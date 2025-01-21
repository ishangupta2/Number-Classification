from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    d = X.shape[1]
    # update bias
    diff = (X @ weight + bias) - y
    b = bias - 2.0 * eta * np.sum(diff)
    w = weight - 2.0 * eta * (X.T @ diff)
    

    for k in range(d):
        # based on lambda bounds, update weight accordingly
        if (w[k] < -2.0 * eta * _lambda):
            w[k] += 2.0 * eta * _lambda
        elif (w[k] > 2.0 * eta * _lambda):
            w[k] -= 2.0 * eta * _lambda
        else:
            w[k] = 0.0

    # return updated weights and bias
    return w, b


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    # calculate regularized SSE Loss 
    return np.linalg.norm((X @ weight + bias) - y, 2) ** 2 + _lambda * np.linalg.norm(weight, 1)


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = start_weight
    old_b: float = start_bias

    new_w, new_b = step(X, y, start_weight, start_bias, _lambda, eta)
    
    while (convergence_criterion(new_w, old_w, new_b, old_b, convergence_delta)):
        # store old weight and old bias
        old_w = np.copy(new_w)
        old_b = np.copy(new_b)
        # update weight and bias for next iteration
        new_w, new_b = step(X, y, old_w, old_b, _lambda, eta)

    return new_w, new_b


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    # calculate infinite norms (max of absolute values) of weight and bias differences
    max_change_w = np.linalg.norm(weight - old_w, np.inf)
    max_change_b = np.abs(bias - old_b)

    if (max_change_w > convergence_delta or max_change_b > convergence_delta):
        return False
    return True


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    sigma = 1.0
    n = 500
    d = 1000
    k = 100

    # Random gaussian noise
    epsilon = np.random.normal(0, sigma ** 2, size = n)

    x = np.random.normal(0, sigma ** 2, size = (n, d))
    x = (x - np.mean(x, axis = 0)) / np.std(x, axis = 0)
    w = np.zeros(d)
    y = x @ w + epsilon

    for j in range(k):
        w[j] = (j + 1) / k

    # takes infinite norm (max of abs values) to get max value of lambda
    lambda_max = 2.0 * np.linalg.norm(x.T @ (y - np.mean(y)), np.inf)

    lambdas = []
    all_non_zeros = []
    fdr = []
    tpr = []
    non_zeros = np.count_nonzero(w)
    curr_lambda = lambda_max
    i = 1

    while (non_zeros < 987):
        print("iteration: " + str(i))
        weights = train(x, y, curr_lambda)[0]
        
        non_zeros = np.count_nonzero(weights)
        all_non_zeros.append(non_zeros)

        incorrect = np.count_nonzero(weights[k:])
        correct = np.count_nonzero(weights[:k])

        fdr.append((incorrect / non_zeros) if non_zeros != 0 else 0)
        tpr.append(correct / k)

        lambdas.append(curr_lambda)
        curr_lambda /= 2.0
        i += 1

    # Plot A
    plt.figure(figsize = (10, 5))
    plt.plot(lambdas, all_non_zeros)
    plt.xscale('log')
    plt.title("Non-zero weights vs. Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("# of non-zero weights")
    plt.show()

    # Plot B
    plt.figure(figsize = (10, 5))
    plt.plot(fdr, tpr)
    plt.title("TPR (True Positive Rate) vs FPR (False Positive Rate)")
    plt.xlabel("FDR")
    plt.ylabel("TPR")
    plt.show()


if __name__ == "__main__":
    main()
