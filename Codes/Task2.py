import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def init_params(layer_sizes, seed):
    np.random.seed(seed)
    params = {}
    for i in range(len(layer_sizes) - 1):
        params["W" + str(i + 1)] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.5
        params["b" + str(i + 1)] = np.zeros(layer_sizes[i + 1])
    return params

def forward(X, params, hidden_act=relu, verbose=True):
    n_layers = len(params) // 2
    cache = {"A0": X}
    A = X
    if verbose:
        print("Input layer:", A.shape)
    for i in range(1, n_layers):
        Z = A @ params["W" + str(i)] + params["b" + str(i)]
        A = hidden_act(Z)
        cache["Z" + str(i)] = Z
        cache["A" + str(i)] = A
        if verbose:
            print("Hidden layer", i, ":", A.shape)
    Z = A @ params["W" + str(n_layers)] + params["b" + str(n_layers)]
    A = softmax(Z)
    cache["Z" + str(n_layers)] = Z
    cache["A" + str(n_layers)] = A
    if verbose:
        print("Output layer:", A.shape)
    return A, cache

layer_sizes = [3, 4, 3, 2]
np.random.seed(7)
X = np.random.rand(6, 3)

params = init_params(layer_sizes, 7)
formula_total = 0
for i in range(len(layer_sizes) - 1):
    W = params["W" + str(i + 1)]
    b = params["b" + str(i + 1)]
    n, m = layer_sizes[i], layer_sizes[i + 1]
    formula_total += n * m + m
    print("Layer", i + 1, ": W =", W.shape,)
    print("b:",b.shape)
    print("params:", n * m + m)
code_total = sum(v.size for v in params.values())
print("Total parameters (formula):", formula_total)
print("Total parameters (code):", code_total)

print("\nForward pass with ReLU hidden layers")
probs, cache = forward(X, params, relu)
print(probs.round(3))
print("Predicted classes:", probs.argmax(axis=1))
print("Row sums:", probs.sum(axis=1).round(3))
print("All rows sum to 1:", np.allclose(probs.sum(axis=1), 1))

print("\nForward pass with Sigmoid hidden layers")
probs_s, cache_s = forward(X, params, sigmoid)
print(probs_s.round(3))
print("Predicted classes:", probs_s.argmax(axis=1))
print("Row sums:", probs_s.sum(axis=1).round(3))

zeros_relu = sum(int(np.sum(cache["A" + str(i)] == 0)) for i in [1, 2])
zeros_sig = sum(int(np.sum(cache_s["A" + str(i)] == 0)) for i in [1, 2])
total_hidden = sum(cache["A" + str(i)].size for i in [1, 2])
print("\nHidden values that are exactly zero (out of", total_hidden, ")")
print("ReLU:", zeros_relu)
print("Sigmoid:", zeros_sig)
print("\nExplanation:")
print("ReLU outputs exactly 0 for every negative input, so some hidden neurons are completely switched off.")
print("Sigmoid always returns a value strictly between 0 and 1, so no hidden value is ever exactly zero.")
print("This makes ReLU activations sparse, while sigmoid activations are always dense.")