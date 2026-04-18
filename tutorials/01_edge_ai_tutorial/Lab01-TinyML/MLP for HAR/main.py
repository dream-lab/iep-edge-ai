import time
import math
from machine import Pin, SPI, LED, I2C
from lsm6dsox import LSM6DSOX

# ---------------------------------------------------------------
# 1. CONFIGURATION & IMPORTS
# ---------------------------------------------------------------
try:
    import mlp_params
except ImportError:
    print("Error: mlp_params.py not found. Please upload it to the board.")
    raise

# Hardware Setup
spi = SPI(5)
cs = Pin("PF6", Pin.OUT_PP, Pin.PULL_UP)
lsm = LSM6DSOX(spi, cs)

# LEDs
led_red = LED("LED_RED")
led_green = LED("LED_GREEN")
led_blue = LED("LED_BLUE")

# Extract Model Parameters
params = mlp_params.nn_params
W1 = params["W1"]
b1 = params["b1"]
W2 = params["W2"]
b2 = params["b2"]
norm_params = mlp_params.norm_params
mean = norm_params["mean"]
scale = norm_params["scale"]

LABELS = ["Sitting", "Flick", "Up-Down"]
WINDOW_SIZE = 20  # Must match the training window size

# ---------------------------------------------------------------
# 2. MATH UTILITIES (The "Old-School" Way)
# ---------------------------------------------------------------
def relu(x):
    """Simple ReLU activation."""
    return x if x > 0 else 0

def softmax(z):
    """Compute softmax probabilities."""
    m = max(z)  # Max for numerical stability
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [e / s for e in exps]

def standardize(features):
    """Scale features: (x - mean) / scale"""
    scaled = []
    for i in range(len(features)):
        val = (features[i] - mean[i]) / scale[i]
        scaled.append(val)
    return scaled

# ---------------------------------------------------------------
# 3. FEATURE EXTRACTION
# ---------------------------------------------------------------
def extract_features(buf):
    """
    Calculates Mean and Std Dev for each axis in the buffer.
    Input: Buffer of [ax, ay, az, gx, gy, gz]
    Output: List of 12 features (6 means, 6 stds)
    """
    n = len(buf)
    num_axes = 6
    features = []

    # 1. Calculate Means
    means = [0.0] * num_axes
    for sample in buf:
        for i in range(num_axes):
            means[i] += sample[i]

    # Divide by N
    for i in range(num_axes):
        means[i] /= n

    # 2. Calculate Variances -> Std Dev
    variances = [0.0] * num_axes
    for sample in buf:
        for i in range(num_axes):
            diff = sample[i] - means[i]
            variances[i] += diff * diff

    stds = []
    for i in range(num_axes):
        stds.append(math.sqrt(variances[i] / n))

    # Combine into single feature vector
    features.extend(means)
    features.extend(stds)

    return features

# ---------------------------------------------------------------
# 4. NEURAL NETWORK INFERENCE (Manual Forward Pass)
# ---------------------------------------------------------------
def mlp_infer(x):
    """
    Executes the MLP forward pass: Output = Softmax(W2 * ReLU(W1 * x + b1) + b2)
    """
    # 1. Preprocess
    x = standardize(x)

    # 2. Hidden Layer (Dense + ReLU)
    h = []
    for j in range(len(b1)):
        # Start with bias
        s = b1[j]
        # Dot product (row of weights * input vector)
        for i in range(len(x)):
            s += x[i] * W1[i][j]
        h.append(relu(s))

    # 3. Output Layer (Dense + Softmax)
    o = []
    for k in range(len(b2)):
        # Start with bias
        s = b2[k]
        # Dot product (row of weights * hidden vector)
        for j in range(len(h)):
            s += h[j] * W2[j][k]
        o.append(s)

    return softmax(o)

def read_imu():
    """Reads 6-axis data from sensor."""
    return list(lsm.accel() + lsm.gyro())

# ---------------------------------------------------------------
# 5. MAIN APPLICATION LOOP
# ---------------------------------------------------------------
print("Nicla Vision: Feature-Based MLP HAR Running...")
print(f"Collecting {WINDOW_SIZE} samples per inference...")

buffer = []

try:
    while True:
        # 1. Fill Buffer
        buffer.clear()
        for _ in range(WINDOW_SIZE):
            buffer.append(read_imu())
            # Sampling delay (adjust based on training rate, e.g. 50Hz)
            time.sleep(0.02)

        # 2. Extract Features (Mean/Std)
        feats = extract_features(buffer)

        # 3. Run Inference
        probs = mlp_infer(feats)
        pred_idx = probs.index(max(probs))
        activity = LABELS[pred_idx]
        confidence = probs[pred_idx]

        # 4. LED Feedback
        led_red.off(); led_green.off(); led_blue.off()
        if pred_idx == 0: led_green.on() # Sitting
        elif pred_idx == 1: led_red.on()  # Flick
        elif pred_idx == 2: led_blue.on() # Up-Down

        # 5. Output
        print(f"Act: {activity:<10} | Conf: {confidence:.2f} | Probs: {[round(p,2) for p in probs]}")

except KeyboardInterrupt:
    print("Stopped.")
