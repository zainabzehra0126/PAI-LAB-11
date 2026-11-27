import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

data = load_digits()
print("shape:", data.data.shape)
print("classes:", data.target_names.tolist())

fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
for ax, img, lab in zip(axes, data.images[:5], data.target[:5]):
    ax.imshow(img, cmap="binary")
    ax.set_title(f"digit {lab}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

results = []
fitted = {}
for arch in [(10,), (64,), (128, 64), (256, 128, 64)]:
    pipe = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=arch, max_iter=500, random_state=42))
    start = time.time()
    pipe.fit(X_tr, y_tr)
    took = time.time() - start
    net = pipe[-1]
    n_params = sum(map(np.size, net.coefs_)) + sum(map(np.size, net.intercepts_))
    results.append((arch, pipe.score(X_te, y_te), n_params, took))
    fitted[arch] = pipe

for arch, acc, n, t in results:
    print(arch)
    print("acc =", round(acc, 3))
    print("params =", n)
    print("time =", round(t, 2), "s")
    print("\n")

pipe = fitted[(128, 64)]
net = pipe[-1]
for i, (W, b) in enumerate(zip(net.coefs_, net.intercepts_), 1):
    print(f"W{i}", W.shape)
    print(f"b{i}", b.shape)
print(" -> ".join(map(str, [64] + [W.shape[1] for W in net.coefs_])))

scaled = pipe[0].transform(X_te[:5])
a = scaled
for W, b in zip(net.coefs_[:-1], net.intercepts_[:-1]):
    a = (a @ W + b).clip(min=0)
z = a @ net.coefs_[-1] + net.intercepts_[-1]
e = np.exp(z - z.max(axis=1, keepdims=True))
mine = e / e.sum(axis=1, keepdims=True)
print(mine.argmax(axis=1), y_te[:5])
print("same as predict_proba:", np.allclose(mine, net.predict_proba(scaled)))