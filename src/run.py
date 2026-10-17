import os

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from data_loading import load_or_build
from features import extract_batch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

X_train, y_train, X_test, y_test = load_or_build()
print("train", X_train.shape, "test", X_test.shape)

X_train_feat = extract_batch(X_train)
X_test_feat = extract_batch(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_feat)
X_test_scaled = scaler.transform(X_test_feat)

svm = SVC(kernel="rbf", C=10, gamma="scale")
svm.fit(X_train_scaled, y_train)

pred = svm.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")

line = f"clean accuracy: {acc:.4f}  macro_f1: {f1:.4f}"
print(line)

out_path = os.path.join(RESULTS_DIR, "results.txt")
with open(out_path, "w") as f:
    f.write(f"train {X_train.shape} test {X_test.shape}\n")
    f.write(line + "\n")

print(f"saved {out_path}")
