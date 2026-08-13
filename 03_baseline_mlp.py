"""
Baseline Experiment: Plain MLP (No Causal Layer)
Goal: Compare accuracy to prove the value of Causal XAI is in Explainability, not just Accuracy.
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob

print("✅ Libraries Imported.")

DATA_PATH = "data/"
TARGET_FILE_PATTERN = "MSSQL.csv"

def main():
    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    target_file = next((f for f in files if TARGET_FILE_PATTERN in f), None)
    if not target_file and files: target_file = files[0]
    if not target_file:
        print("❌ Error: Data not found in 'data/' folder.")
        return

    print(f"📂 Loading: {target_file}")
    df = pd.read_csv(target_file, nrows=50000, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    y_raw = (df['Label'].astype(str).str.upper() != 'BENIGN').astype(int).values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('Label', errors='ignore')
    top_4_cols = df[numeric_cols].var().nlargest(4).index.tolist()
    X_raw = df[top_4_cols].values
    
    idx_b = np.where(y_raw == 0)[0]
    idx_a = np.where(y_raw == 1)[0]
    n = min(len(idx_b), len(idx_a), 20000)
    idx_bal = np.concatenate([np.random.choice(idx_b, n), np.random.choice(idx_a, n)])
    np.random.shuffle(idx_bal)
    
    X_scaled = StandardScaler().fit_transform(X_raw[idx_bal])
    y_bal = y_raw[idx_bal]
    
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_bal, test_size=0.2)

    # Plain MLP (No Causal Layer)
    inputs = layers.Input(shape=(4,))
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\n🚀 Training Baseline MLP...")
    model.fit(X_tr, y_tr, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n📈 BASELINE Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()