// Includes based on Harvard_TinyMLx magic_wand.ino example
#include <TensorFlowLite.h> // General library header
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h" // For TFLITE_SCHEMA_VERSION
#include "tensorflow/lite/version.h"                 // For TFLITE_VERSION_STRING (optional but good)
#include <tensorflow/lite/c/common.h>               // For TfLiteTensor, kTfLiteOk

#include <Arduino_LSM9DS1.h> // IMU library for Nano 33 BLE Sense
#include "model_data.h"      // YOUR TFLite model data (should contain gesture_model_tflite array)

// Constants from your Python script for your gesture model
const int GESTURE_SAMPLES = 119; // Sequence length
const int NUM_FEATURES = 6;      // ax, ay, az, gx, gy, gz
const int NUM_GESTURES = 4;      // Number of classes (0:wave, 1:uppercut, 2:fist_bump, 3:circle)

// Normalization factors - MUST MATCH YOUR PYTHON SCRIPT
const float ACCEL_NORMALIZATION_FACTOR = 4.0f;
const float GYRO_NORMALIZATION_FACTOR = 2000.0f;

// Anonymous namespace for TFLM objects, similar to magic_wand example
namespace {

// TensorFlow Lite global objects
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr; // Renamed to avoid conflict if 'input' is a keyword/macro
TfLiteTensor* output_tensor = nullptr;// Renamed

// Tensor Arena: Memory for TFLM operations. Adjust size if allocation fails.
// Your float32 model with Conv layers might need a decent amount.
constexpr int kTensorArenaSize = 20 * 1024; // Start with 20KB, adjust as needed
uint8_t tensor_arena[kTensorArenaSize];

// Buffer to store one gesture sequence
float gesture_buffer[GESTURE_SAMPLES * NUM_FEATURES];
int gesture_buffer_idx = 0;
bool new_gesture_data_ready = false;

// Emoji mapping for your gestures
const char* emoji_map[NUM_GESTURES] = {
  "👋 Wave",
  "🥊 Uppercut",
  "👊 Fist Bump",
  "🔄 Circle"
};

} // anonymous namespace

// --- SETUP ---
void setup() {
  Serial.begin(9600);
  // Optional: wait for serial connection for debugging
  // while (!Serial) { delay(100); }

  // 1. Initialize Error Reporter (following magic_wand.ino pattern)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  error_reporter->Report("Gesture Recognition Sketch Started");

  // 2. Initialize IMU
  if (!IMU.begin()) {
    error_reporter->Report("ERROR: Failed to initialize IMU!");
    while (1); // Halt
  }
  // IMPORTANT: Set IMU Full Scale ranges to match your normalization assumptions
  // These functions might vary slightly based on the exact LSM9DS1 library version.
  // Consult your specific Arduino_LSM9DS1 library if these are incorrect.
  // Assuming the library default for Nano 33 BLE Sense often covers typical ranges,
  // but explicit setting is safer.
  // Example: IMU.setAccelFS(4); // Set Accelerometer Full Scale to +/- 4G
  // Example: IMU.setGyroFS(2000); // Set Gyroscope Full Scale to +/- 2000 dps
  error_reporter->Report("IMU initialized");

  // 3. Load the TensorFlow Lite Model
  // Make sure 'gesture_model_tflite' is the name of the array in your model_data.h
  model = tflite::GetModel(gesture_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("ERROR: Model schema version %d not equal to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    while (1); // Halt
  }
  error_reporter->Report("TFLite model loaded successfully");


  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D(); 
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  // 5. Build the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // 6. Allocate Tensors
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("ERROR: AllocateTensors() failed. Increase kTensorArenaSize?");
    while (1); // Halt
  }
  error_reporter->Report("Tensors allocated");

  // 7. Get Pointers to Input and Output Tensors
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Optional: Verify input tensor details (shape, type) for your float32 model
  // Expected shape for your model: [1, 119, 6] (batch, samples, features)
  // The TFLM interpreter might represent this as 3 dims: [1, 119, 6] or 4 dims: [1, 1, 119, 6] or [1, 119, 6, 1]
  // Check your model's input details if issues arise.
  error_reporter->Report("Input tensor details:");
  error_reporter->Report("Dimensions: %d", input_tensor->dims->size);
  for (int i = 0; i < input_tensor->dims->size; ++i) {
    error_reporter->Report("Dim %d size: %d", i, input_tensor->dims->data[i]);
  }
  error_reporter->Report("Type: %s", TfLiteTypeGetName(input_tensor->type)); // Should be kTfLiteFloat32

  error_reporter->Report("Setup complete. Ready to detect gestures.");
  Serial.println("Perform a gesture...");
}

// --- LOOP ---
void loop() {
  float ax, ay, az, gx, gy, gz;

  // Check if IMU data is available
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax, ay, az);
    IMU.readGyroscope(gx, gy, gz);

    // --- CRITICAL: Normalize data exactly as in your Python script ---
    ax /= ACCEL_NORMALIZATION_FACTOR;
    ay /= ACCEL_NORMALIZATION_FACTOR;
    az /= ACCEL_NORMALIZATION_FACTOR;
    gx /= GYRO_NORMALIZATION_FACTOR;
    gy /= GYRO_NORMALIZATION_FACTOR;
    gz /= GYRO_NORMALIZATION_FACTOR;

    // Store in buffer until we have GESTURE_SAMPLES
    if (gesture_buffer_idx < GESTURE_SAMPLES * NUM_FEATURES) {
      gesture_buffer[gesture_buffer_idx++] = ax;
      gesture_buffer[gesture_buffer_idx++] = ay;
      gesture_buffer[gesture_buffer_idx++] = az;
      gesture_buffer[gesture_buffer_idx++] = gx;
      gesture_buffer[gesture_buffer_idx++] = gy;
      gesture_buffer[gesture_buffer_idx++] = gz;

      if (gesture_buffer_idx == GESTURE_SAMPLES * NUM_FEATURES) {
        new_gesture_data_ready = true;
        // Buffer is full, ready for inference
      }
    }
  }

  // If a full gesture sequence is ready for inference
  if (new_gesture_data_ready) {
    error_reporter->Report("New gesture data ready. Running inference...");

    // Copy data from our gesture_buffer to the model's input tensor
    // The input tensor expects float32 data.
    for (int i = 0; i < GESTURE_SAMPLES * NUM_FEATURES; ++i) {
      input_tensor->data.f[i] = gesture_buffer[i]; // .f for float data
    }

    // Run inference
    unsigned long invoke_start_time = millis();
    if (interpreter->Invoke() != kTfLiteOk) {
      error_reporter->Report("ERROR: Invoke() failed!");
      new_gesture_data_ready = false; // Reset flag
      gesture_buffer_idx = 0;         // Reset buffer index
      return;
    }
    unsigned long invoke_end_time = millis();
    error_reporter->Report("Inference time: %d ms", (invoke_end_time - invoke_start_time));

    // Get the output tensor (probabilities for each gesture class)
    // output_tensor->data.f will point to the float array of probabilities
    int predicted_class = -1;
    float max_probability = 0.0f;

    Serial.print("Output Probabilities: [");
    for (int i = 0; i < NUM_GESTURES; ++i) {
      float current_prob = output_tensor->data.f[i];
      Serial.print(current_prob, 4); // Print with 4 decimal places
      if (current_prob > max_probability) {
        max_probability = current_prob;
        predicted_class = i;
      }
      if (i < NUM_GESTURES - 1) Serial.print(", ");
    }
    Serial.println("]");

    // Display the predicted gesture
    if (predicted_class != -1) {
      Serial.print("==> Predicted Gesture: ");
      Serial.print(emoji_map[predicted_class]);
      Serial.print(" (Class: ");
      Serial.print(predicted_class);
      Serial.print(", Confidence: ");
      Serial.print(max_probability * 100.0f);
      Serial.println("%)");
    } else {
      Serial.println("Could not classify gesture with confidence.");
    }

    new_gesture_data_ready = false; // Reset flag, ready for next gesture
    gesture_buffer_idx = 0;         // Reset buffer index
    Serial.println("\nPerform another gesture...");

    // Optional: Add a small delay to prevent immediate re-triggering
    // delay(500);
  }
}