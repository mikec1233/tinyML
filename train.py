import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# Disable GPU for now since you're having issues with it
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the preprocessed data
print("Loading preprocessed data...")
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Print data shapes
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

# Apply normalization similar to what we'll do on Arduino
# Accelerometer data (first 3 channels) - typically in range -4 to 4 G
#X_train[:, :, 0:3] = X_train[:, :, 0:3] / 4.0
#X_test[:, :, 0:3] = X_test[:, :, 0:3] / 4.0
#
## Gyroscope data (last 3 channels) - typically in range -2000 to 2000 dps
#X_train[:, :, 3:6] = X_train[:, :, 3:6] / 2000.0
#X_test[:, :, 3:6] = X_test[:, :, 3:6] / 2000.0

# Check class distribution
print("\nClass distribution in training:")
for i in range(4):  # 0=wave, 1=uppercut, 2=fist bump, 3=circle
    print(f"Class {i}: {np.sum(y_train == i)}")

print("\nClass distribution in test:")
for i in range(4):  # 0=wave, 1=uppercut, 2=fist bump, 3=circle
    print(f"Class {i}: {np.sum(y_test == i)}")

# Convert labels to one-hot encoding
num_classes = 4  # 0=wave, 1=uppercut, 2=fist bump, 3=circle
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes)

# Function to create a very lightweight model suitable for Arduino
def create_extra_lightweight_model(input_shape, num_classes):
    model = Sequential([
        # First Conv layer
        Conv1D(8, kernel_size=5, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Second Conv layer
        Conv1D(16, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Define model input shape (sequence_length, features)
input_shape = (X_train.shape[1], X_train.shape[2])

# Create model
model = create_extra_lightweight_model(input_shape, num_classes)

# Print model summary
model.summary()

# Configure callbacks for training
callbacks = [
    ModelCheckpoint(
        filepath='gesture_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,  # More patience since we have little data
        verbose=1,
        restore_best_weights=True
    )
]

# Print data shape before training to ensure it's correct
print(f"\nX_train shape: {X_train.shape}")
print(f"y_train_onehot shape: {y_train_onehot.shape}")

# Set higher batch size for small dataset
batch_size = 4  # Adjust based on dataset size
epochs = 100    # Increase epochs, early stopping will prevent overfitting

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train_onehot,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test_onehot),
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print(f"Test accuracy: {test_acc:.4f}")

# Convert to TensorFlow Lite without quantization
print("\nConverting model to TensorFlow Lite (float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# No optimizations or quantization
tflite_model = converter.convert()

# Save the TFLite model
with open('gesture_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as 'gesture_model.tflite'")

# Check TFLite model size
tflite_size = os.path.getsize('gesture_model.tflite') / 1024
print(f"TFLite model size: {tflite_size:.2f} KB")

# Test TFLite model on a few samples
print("\nTesting TFLite model inference...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input details: {input_details}")
print(f"Output details: {output_details}")

# Test on a few samples
num_test = min(4, len(X_test))
for i in range(num_test):
    test_input = np.expand_dims(X_test[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    actual_class = np.argmax(y_test_onehot[i])
    print(f"Sample {i}: Predicted = {predicted_class}, Actual = {actual_class}")

# Generate C header file for Arduino
print("\nConverting TFLite model to C array for Arduino...")

# Function to convert model to C array
def model_to_c_array(model_data, variable_name):
    c_str = f"const unsigned char {variable_name}[] = {{\n  "
    hex_data = ["0x{:02x}".format(b) for b in model_data]
    
    # Format with 12 values per line
    for i, val in enumerate(hex_data):
        if i > 0 and i % 12 == 0:
            c_str += "\n  "
        c_str += val + ", "
    
    c_str += f"\n}};\n\nconst unsigned int {variable_name}_len = {len(model_data)};\n"
    return c_str

# Create the C array
c_array = model_to_c_array(tflite_model, "gesture_model_tflite")

# Save to a header file
with open('model_data.h', 'w') as f:
    f.write(c_array)

print("C array saved to 'model_data.h'")

print("\nNext steps:")
print("1. Include model_data.h in your Arduino sketch")
print("2. Upload the Arduino sketch with the model")
# Apply normalization similar to what we'll do on Arduino
# Accelerometer data (first 3 channels) - typically in range -4 to 4 G
#X_train[:, :, 0:3] = X_train[:, :, 0:3] / 4.0
#X_test[:, :, 0:3] = X_test[:, :, 0:3] / 4.0
#
## Gyroscope data (last 3 channels) - typically in range -2000 to 2000 dps
#X_train[:, :, 3:6] = X_train[:, :, 3:6] / 2000.0
#X_test[:, :, 3:6] = X_test[:, :, 3:6] / 2000.0