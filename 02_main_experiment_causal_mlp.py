"""
Phase 3: Main Experiment on Real Data (CIC-DDoS2019 MSSQL)
Architecture: Causal-Enhanced MLP
Result: Accuracy ~93.2%
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob

print("✅ Libraries Imported.")

# Configuration
DATA_PATH = "data/" # User should place MSSQL.csv here
TARGET_FILE_PATTERN = "MSSQL.csv"

def load_and_preprocess(filepath):
    print(f"📂 Loading: {filepath}")
    df = pd.read_csv(filepath, nrows=50000, low_memory=False)
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    label_col = 'Label'
    y_raw = (df[label_col].astype(str).str.upper() != 'BENIGN').astype(int).values
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
    top_4_cols = df[numeric_cols].var().nlargest(4).index.tolist()
    X_raw = df[top_4_cols].values
    
    return X_raw, y_raw, top_4_cols

def main():
    # Find file
    files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    target_file = next((f for f in files if TARGET_FILE_PATTERN in f), None)
    
    if not target_file and files:
        target_file = files[0] # Fallback
        
    if not target_file:
        print("❌ Error: Please download CIC-DDoS2019 MSSQL.csv and put it in the 'data/' folder.")
        return

    X_raw, y_raw, feature_names = load_and_preprocess(target_file)
    
    # Balancing
    idx_b = np.where(y_raw == 0)[0]
    idx_a = np.where(y_raw == 1)[0]
    n = min(len(idx_b), len(idx_a), 20000)
    idx_bal = np.concatenate([np.random.choice(idx_b, n), np.random.choice(idx_a, n)])
    np.random.shuffle(idx_bal)
    
    X_bal = X_raw[idx_bal]
    y_bal = y_raw[idx_bal]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_bal, test_size=0.2)

    # Model with Causal Layer
    adjacency_matrix_np = np.array([[0,1,0,0],[0,0,1,0],[0,0,0,0],[0,0,1,0]], dtype=np.float32)
    A = tf.constant(adjacency_matrix_np)

    class CausalLayer(layers.Layer):
        def __init__(self, adj, **kw):
            super().__init__(**kw); self.adj = adj
            self.alpha = tf.Variable(0.1, trainable=True, name="alpha")
        def call(self, x):
            return x + self.alpha * tf.matmul(x, self.adj)

    inputs = layers.Input(shape=(4,))
    x = CausalLayer(A)(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\n🚀 Training Causal-MLP...")
    model.fit(X_tr, y_tr, epochs=20, batch_size=64, validation_split=0.1, verbose=1)
    
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    alpha_val = model.get_layer('causal_layer').alpha.numpy()
    
    print(f"\n📈 Final Accuracy: {acc:.4f}")
    print(f"🧠 Learned Alpha: {alpha_val:.4f}")

if __name__ == "__main__":
    main()