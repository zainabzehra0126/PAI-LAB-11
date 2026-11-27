import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

inputs = list(itertools.product([0, 1], repeat=2))
X = np.array(inputs)
gates = {"OR": [a | b for a, b in inputs], "NAND": [1 - (a & b) for a, b in inputs], "XOR": [a ^ b for a, b in inputs]}

def fire(z):
    return int(z >= 0)

def perceptron(X, y, lr=0.1, epochs=30):
    w, b = np.zeros(2), 0.0
    for _ in range(epochs):
        for xi, t in zip(X, y):
            e = t - fire(w @ xi + b)
            w, b = w + lr * e * xi, b + lr * e
    return w, b

for name, y in gates.items():
    w, b = perceptron(X, y)
    got = [fire(w @ xi + b) for xi in X]
    print(f"{name:5s} got {got} want {y} -> {'Ok' if got == y else 'Failed'}")

hidden_w = np.array([[2.0, -2.0], [2.0, -2.0]])
hidden_b = np.array([-1.0, 3.0])
out_w = np.array([1.0, 1.0])
out_b = -1.5

print("\nx1 x2 : h1 h2 : y")
hidden_points = []
for xi in X:
    h = [fire(v) for v in xi @ hidden_w + hidden_b]
    y_out = fire(out_w @ np.array(h) + out_b)
    hidden_points.append(h)
    print(f" {xi[0]}  {xi[1]} |  {h[0]}  {h[1]} | {y_out}")
hidden_points = np.array(hidden_points)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, pts, labels in zip(axes, [X, hidden_points], [("x1", "x2"), ("h1", "h2")]):
    ax.scatter(pts[:, 0], pts[:, 1], c=gates["XOR"], cmap="coolwarm", s=200, edgecolor="black")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.3)
axes[0].set_title("Input space")
axes[1].set_title("Hidden space")
plt.show()

print("\nThe hidden layer folds the two 1-cases onto one point, so a single line can split 1s from 0s in hidden space.")

for size in (2, 4):
    wins = [np.array_equal(MLPClassifier(hidden_layer_sizes=(size,), activation="tanh", solver="lbfgs", max_iter=1000, random_state=s).fit(X, gates["XOR"]).predict(X), gates["XOR"]) for s in range(10)]
    print(f"size {size}: solved {sum(wins)}/10 :", "".join("Y" if w else "n" for w in wins))
print("More hidden neurons make it less likely that training gets stuck in a bad local solution.")
