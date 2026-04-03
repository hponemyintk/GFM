"""Quick smoke test for TabPFN v2 classification and regression."""
import os
os.environ["TABPFN_MODEL_VERSION"] = "v2"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor

rng = np.random.RandomState(42)
X_train = rng.randn(200, 504).astype(np.float32)  # matches 3*168 feature dim
X_test = rng.randn(50, 504).astype(np.float32)

# Classification
y_cls = (X_train[:, 0] > 0).astype(int)
clf = TabPFNClassifier(device="cuda", n_estimators=1)
clf.fit(X_train, y_cls)
probs = clf.predict_proba(X_test)
print(f"Classification: predict_proba shape={probs.shape}, range=[{probs.min():.3f}, {probs.max():.3f}]")

# Regression
y_reg = X_train[:, 0] + rng.randn(200).astype(np.float32) * 0.1
reg = TabPFNRegressor(device="cuda", n_estimators=1)
reg.fit(X_train, y_reg)
preds = reg.predict(X_test)
print(f"Regression:     predict shape={preds.shape}, range=[{preds.min():.3f}, {preds.max():.3f}]")

print("\nTabPFN v2 OK")
