import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the data
print("Loading data...")
df = pd.read_csv('gesture_data.csv')

# Quick overview of the data
print(f"Dataset shape: {df.shape}")
print("\nClass distribution:")
print(df['label'].value_counts())

# Reshape data for ML model - each gesture is a sequence of 119 samples
num_gestures = len(df) // 119
print(f"\nNumber of gesture sequences: {num_gestures}")

# Reshape features into sequences
features = df.iloc[:, :-1].values  # All columns except label
labels = df.iloc[:, -1].values     # Just the label column

# Reshape to [num_gestures, sequence_length, num_features]
features = features.reshape(num_gestures, 119, 6)

# Normalize the data
# Accelerometer data (first 3 channels) - typically in range -4 to 4 G
features[:, :, 0:3] = features[:, :, 0:3] / 4.0

# Gyroscope data (last 3 channels) - typically in range -2000 to 2000 dps
features[:, :, 3:6] = features[:, :, 3:6] / 2000.0

# For labels, we just need one label per sequence
labels = np.array([labels[i*119] for i in range(num_gestures)])

print(f"\nFeatures shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Check class distribution in the gesture sequences
unique_labels, counts = np.unique(labels, return_counts=True)
print("\nGesture sequence class distribution:")
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count}")

# Split into training and test sets using STRATIFIED sampling
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} sequences")
print(f"Test set: {X_test.shape[0]} sequences")

# Check class distribution in training and test sets
print("\nTraining set class distribution:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for label, count in zip(unique_train, counts_train):
    print(f"Class {label}: {count}")

print("\nTest set class distribution:")
unique_test, counts_test = np.unique(y_test, return_counts=True)
for label, count in zip(unique_test, counts_test):
    print(f"Class {label}: {count}")

# Save processed data for model training
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("\nProcessed data saved to .npy files")

# Optional: Visualize one sample from each class
plt.figure(figsize=(15, 10))
class_samples = {}

# Find one sample from each class in the training set
for i, label in enumerate(y_train):
    if label not in class_samples:
        class_samples[label] = i
    if len(class_samples) == 4:  # All classes found
        break

# Plot samples
for class_label, sample_idx in class_samples.items():
    gesture = X_train[sample_idx]
    
    # Plot accelerometer data (already normalized)
    plt.subplot(4, 2, class_label*2 + 1)
    plt.plot(gesture[:, 0], label='aX')
    plt.plot(gesture[:, 1], label='aY')
    plt.plot(gesture[:, 2], label='aZ')
    plt.title(f'Class {class_label} - Accelerometer (Normalized)')
    plt.legend()
    
    # Plot gyroscope data (already normalized)
    plt.subplot(4, 2, class_label*2 + 2)
    plt.plot(gesture[:, 3], label='gX')
    plt.plot(gesture[:, 4], label='gY')
    plt.plot(gesture[:, 5], label='gZ')
    plt.title(f'Class {class_label} - Gyroscope (Normalized)')
    plt.legend()

plt.tight_layout()
plt.savefig('gesture_samples_normalized.png')
print("Sample visualizations saved to 'gesture_samples_normalized.png'")