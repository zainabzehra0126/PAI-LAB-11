import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.6, -0.3, 0.8, 0.1])
w = np.array([0.5, 0.9, -0.4, 0.7])
b = -0.2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def neuron(x, w, b, activation):
    z = np.dot(w, x) + b
    if activation == "sigmoid":
        a = sigmoid(z)
    elif activation == "tanh":
        a = tanh(z)
    elif activation == "relu":
        a = relu(z)
    return z, a

for act in ["sigmoid", "tanh", "relu"]:
    z, a = neuron(x, w, b, act)
    print(f"{act}: z = {z:.4f}")
    print(f"Output = {a:.4f}")

def d_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def d_tanh(z):
    return 1 - np.tanh(z) ** 2

def d_relu(z):
    return np.where(z > 0, 1, 0)

z_val = np.dot(w, x) + b
print("\nDerivatives at z =", round(z_val, 4))
print("sigmoid'(z):", round(float(d_sigmoid(z_val)), 4))
print("tanh'(z):", round(float(d_tanh(z_val)), 4))
print("relu'(z):", int(d_relu(z_val)))

z = np.linspace(-6, 6, 300)
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label="Sigmoid")
plt.plot(z, tanh(z), label="Tanh")
plt.plot(z, relu(z), label="ReLU")
plt.xlabel("z")
plt.ylabel("f(z)")
plt.title("Activation Functions")
plt.legend()
plt.grid(alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(z, d_sigmoid(z), label="Sigmoid'")
plt.plot(z, d_tanh(z), label="Tanh'")
plt.plot(z, d_relu(z), label="ReLU'")
plt.xlabel("z")
plt.ylabel("f'(z)")
plt.title("Derivatives")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

scores = np.array([3.0, 1.0, 0.2, -1.0])
p = softmax(scores)
print("\nSoftmax:", np.round(p, 4))
print("Sum of probabilities:", round(p.sum(), 4))
p_shift = softmax(scores + 100)
print("Softmax after adding 100:", np.round(p_shift, 4))
print("Same result:", np.allclose(p, p_shift))