"""
FINAL SUBMISSION GENERATOR
--------------------------
Generates the 'submission.csv' file for the Kaggle Leaderboard.

Process:
1. Loads the 'Model' class (Inference Wrapper).
2. Iterates through 'test.csv'.
3. Generates position predictions [0.0, 2.0].
4. Saves to CSV.
"""
import pandas as pd
import numpy as np
from model import Model

print("="*80)
print("CREATING SUBMISSION FILE")
print("="*80)

# Load model
print("\n[1/3] Loading model...")
model = Model()

# Load test data
print("\n[2/3] Loading test data...")
test = pd.read_csv('test.csv')
print(f"  Test data: {test.shape}")

#Generate predictions
print("\n[3/3] Generating predictions...")
predictions = []

for idx in range(len(test)):
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{len(test)} ({idx/len(test)*100:.1f}%)")
    
    test_row = test.iloc[[idx]]
    position = model.predict(test_row, current_holdings=1.0)
    predictions.append(position)

predictions = np.array(predictions)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': range(len(predictions)),
    'position': predictions
})

# Save
output_file = 'submission.csv'
submission.to_csv(output_file, index=False)

print(f"\n✓ Saved {output_file}")
print(f"\nSubmission statistics:")
print(f"  Total predictions: {len(predictions)}")
print(f"  Mean position: {predictions.mean():.4f}")
print(f"  Std position: {predictions.std():.4f}")
print(f"  Min position: {predictions.min():.4f}")
print(f"  Max position: {predictions.max():.4f}")
print(f"  % at 0.0: {(predictions == 0.0).sum() / len(predictions) * 100:.2f}%")
print(f"  % at 2.0: {(predictions == 2.0).sum() / len(predictions) * 100:.2f}%")

# Distribution
print(f"\nPosition distribution:")
for pos_val in [0.0, 0.5, 1.0, 1.5, 2.0]:
    count = (np.abs(predictions - pos_val) < 0.01).sum()
    pct = count / len(predictions) * 100
    if count > 0:
        print(f"  {pos_val:.1f}: {count} ({pct:.2f}%)")

print("\n✅ SUBMISSION FILE READY FOR UPLOAD")
print(f"\nUpload {output_file} to Kaggle competition")
