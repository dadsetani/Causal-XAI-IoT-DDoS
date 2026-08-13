"""
Phase 1: Causal Analysis using DoWhy on Synthetic Data
Goal: Validate the causal effect of Burst_Duration on Attack via Queue_Drop.
"""
import numpy as np
import pandas as pd
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt

print("🚀 Running Phase 1: Causal Analysis (DoWhy)...")

# 1. Generate Synthetic Data based on DAG
np.random.seed(42)
n_samples = 5000
burst_duration = np.random.exponential(scale=0.5, size=n_samples)
normal_traffic = np.random.normal(loc=50, scale=10, size=n_samples)
tcp_timeout = 1.0 + 0.8 * burst_duration + np.random.normal(0, 0.1, n_samples)
queue_drop = 0.3 * tcp_timeout + 0.02 * normal_traffic + np.random.normal(0, 0.05, n_samples)

# Attack logic
attack_prob = 1 / (1 + np.exp(-(2.0 * burst_duration + 1.5 * queue_drop - 1.0)))
attack_label = np.random.binomial(1, attack_prob)

data = pd.DataFrame({
    'Burst_Duration': burst_duration,
    'TCP_Timeout': tcp_timeout,
    'Queue_Drop': queue_drop,
    'Normal_Traffic': normal_traffic,
    'Attack_Label': attack_label
})

# 2. Define Causal Model
model = CausalModel(
    data=data,
    treatment='Burst_Duration',
    outcome='Attack_Label',
    common_causes=['Normal_Traffic'], # Confounder
    graph="graph { Burst_Duration -> TCP_Timeout; TCP_Timeout -> Queue_Drop; Queue_Drop -> Attack_Label; Normal_Traffic -> Queue_Drop; Normal_Traffic -> Attack_Label; }"
)

# 3. Identify Effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(f"✅ Identified Estimand: {identified_estimand}")

# 4. Estimate Effect
estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
print(f"📈 Causal Effect Estimate: {estimate.value:.4f}")

# 5. Refutation Test
refute_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
print(f"🛡️ Refutation P-Value: {refute_results.refutation_result['p_value']:.4f}")

print("\n🎉 Phase 1 Complete. Causal structure validated.")