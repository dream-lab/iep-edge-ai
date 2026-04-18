import time
from machine import Pin, SPI, LED
from lsm6dsox import LSM6DSOX

# -------------------------------
# Hardware setup
# -------------------------------
spi = SPI(5)
cs = Pin("PF6", Pin.OUT_PP, Pin.PULL_UP)
lsm = LSM6DSOX(spi, cs=cs)

red_led = LED("LED_RED")
green_led = LED("LED_GREEN")

# -------------------------------
# Decision tree model
# input = [ax, ay, az]
# -------------------------------
def score(input):
    if input[0] <= 0.6346435248851776:
        if input[2] <= -0.6030884981155396:
            if input[2] <= -0.7907100021839142:
                if input[1] <= -0.38256850838661194:
                    return [1.0, 0.0]
                else:
                    if input[0] <= 0.3996579945087433:
                        return [1.0, 0.0]
                    else:
                        return [0.56, 0.44]
            else:
                if input[0] <= 0.4990234822034836:
                    if input[1] <= -0.47222900390625:
                        return [0.0, 1.0]
                    else:
                        return [0.26, 0.74]
                else:
                    if input[0] <= 0.5129395127296448:
                        return [1.0, 0.0]
                    else:
                        return [0.5, 0.5]
        else:
            return [0.0, 1.0]
    else:
        return [1.0, 0.0]

# -------------------------------
# Activity mapping
# -------------------------------
activity_map = {
    0: "idle",
    1: "wrist flick"
}

print("Nicla Vision: Right-aligned accel + prediction")

# -------------------------------
# Main loop
# -------------------------------
try:
    while True:
        # Read accelerometer
        ax, ay, az = lsm.accel()
        feature = [ax, ay, az]

        # Run inference
        prediction = score(feature)
        predicted_idx = prediction.index(max(prediction))
        predicted_activity = activity_map[predicted_idx]

        # LED control
        if predicted_idx == 0:
            green_led.on()
            red_led.off()
        else:
            red_led.on()
            green_led.off()

        # Prediction formatting (2 decimal places)
        pred_str = "[{:.2f}, {:.2f}]".format(
            prediction[0], prediction[1]
        )

        # Right-aligned accel print (exact style requested)
        print(
            "Accelerometer: x:{:>8.3f} y:{:>8.3f} z:{:>8.3f} | "
            "Prediction: {} | Pred: {}".format(
                ax, ay, az,
                pred_str,
                predicted_activity
            )
        )

        time.sleep_ms(10)

except KeyboardInterrupt:
    red_led.off()
    green_led.off()
    print("Stopped")
