import asyncio
import time
import numpy as np
from bleak import BleakScanner
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import ttk
import threading
from tkinter import font
import pandas as pd 
from PIL import Image, ImageTk
from tkinter import messagebox
import csv
import math
import time

# Path to your log file
logfile = "/tmp/o"

# Keeps track of the latest readings from each device
latest_readings = {}

# Store the start time to calculate elapsed time
start_time = time.time()

# Initialize NumPy arrays to store time and temperatures
time_array = np.array([])      # Elapsed time in minutes
T_inside = np.array([])        # Inside temperature (°F)
T_outside = np.array([])       # Outside temperature (°F)
H_inside = np.array([])        # Inside humidity (%)
H_outside = np.array([])        # Outside humidity (%)

# Initialize the machine learning model
ml_model = None

# Create the root window for tkinter
root = tk.Tk()
root.title("HeatSync AI - Temperature and Humidity Monitoring")
root.geometry("1200x1000")

# Add labels and frames for displaying device information
device_frame = tk.LabelFrame(root, text="Detected Devices", padx=10, pady=10)
device_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

log_frame = tk.LabelFrame(root, text="Logging Information", padx=10, pady=10)
log_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

prediction_frame = tk.LabelFrame(root, text="ML and Cooling Model Predictions", padx=10, pady=10)
prediction_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

target_temp_frame = tk.LabelFrame(root, text="Target Temperature", padx=10, pady=10)
target_temp_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

# Configure the grid to adjust with window resizing
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure([0, 1, 2, 3], weight=1)

# Text widget to display logs
log_text = tk.Text(log_frame, height=10, state=tk.DISABLED)
log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Labels for predictions
prediction_label = tk.Label(prediction_frame, text="Prediction: ", anchor="w")
prediction_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

future_predictions_label = tk.Label(prediction_frame, text="Future Predictions: ", anchor="w")
future_predictions_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Target temperature label
target_temp_label = tk.Label(target_temp_frame, text="Target : 70.00°F\n", anchor="w", justify="left")
target_temp_label.grid(row=0, column=0, padx=10, pady=10)

# Images for windows
image1 = Image.open('open.jpeg')
image2 = Image.open('closed.jpeg')

max_size = (150, 150)
image1.thumbnail(max_size, Image.LANCZOS)
image2.thumbnail(max_size, Image.LANCZOS)

photo1 = ImageTk.PhotoImage(image1)
photo2 = ImageTk.PhotoImage(image2)

current_image = tk.StringVar(value="image2")

image_label = tk.Label(target_temp_frame, image=photo2)
image_label.grid(row=0, column=1, padx=10, pady=10)

# Images for curtains
curtain_image1 = Image.open('curtain_open.jpeg')
curtain_image2 = Image.open('curtain_closed.jpeg')

curtain_image1.thumbnail(max_size, Image.LANCZOS)
curtain_image2.thumbnail(max_size, Image.LANCZOS)

curtain_photo1 = ImageTk.PhotoImage(curtain_image1)
curtain_photo2 = ImageTk.PhotoImage(curtain_image2)

curtain_current_image = tk.StringVar(value="curtain_image2")

curtain_image_label = tk.Label(target_temp_frame, image=curtain_photo2)
curtain_image_label.grid(row=0, column=2, padx=10, pady=10)

# Images for fans 
fan_image1 = Image.open('fan_open.jpeg')
fan_image2 = Image.open('fan_closed.jpeg')

fan_image1.thumbnail(max_size, Image.LANCZOS)
fan_image2.thumbnail(max_size, Image.LANCZOS)

fan_photo1 = ImageTk.PhotoImage(fan_image1)
fan_photo2 = ImageTk.PhotoImage(fan_image2)

fan_current_image = tk.StringVar(value="fan_image2")

fan_image_label = tk.Label(target_temp_frame, image=fan_photo2)
fan_image_label.grid(row=0, column=3, padx=10, pady=10)

# Function to switch images
def switch_image(img):
    if img == 'closed':
        image_label.config(image=photo2)
        #image_label.image = photo2
        #current_image.set("image2")
    else:
        image_label.config(image=photo1)
        #image_label.image = photo1
        #current_image.set("image1")

def switch_curtain_image(img):
    if img == 'closed':
        curtain_image_label.config(image=curtain_photo2)
        #curtain_image_label.image = curtain_photo2
        #curtain_current_image.set("curtain_image2")
    else:
        curtain_image_label.config(image=curtain_photo1)
        #curtain_image_label.image = curtain_photo1
        #curtain_current_image.set("curtain_image1")

def switch_fan_image(img):
    if img == 'closed':
        fan_image_label.config(image=fan_photo2)
    else:
        fan_image_label.config(image=fan_photo1)

# Define the device info label once, outside of the function
fixed_width_font = font.Font(family="Courier New", size=16)
device_label = tk.Label(device_frame, text="", anchor="w", font=fixed_width_font)
device_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")


inprogress = False
desired_temp = 70 

def detection_callback(device, advertisement_data):
    global latest_readings

    # Check if advertisement_data.local_name exists
    if advertisement_data.local_name is not None:
        # Check if the device is of interest (by name)
        if 'GVH5075' in advertisement_data.local_name or 'GVH' in advertisement_data.local_name:
            try:
                # Extract temperature and humidity data from the manufacturer-specific data
                payload = advertisement_data.manufacturer_data[60552][1:5]
                temphum = int.from_bytes(payload[0:3], "big")
                hum10 = temphum % 1000
                temp = (temphum - hum10) / 10000
                hum = hum10 / 10

                # Convert temperature to Fahrenheit (optional)
                temp = (temp * 9/5) + 32

                # Update the latest readings for the detected device
                latest_readings[device.name] = (temp, hum)

                # Update GUI with latest readings
                update_device_info()

            except (KeyError, IndexError):
                print(f"Error processing data from {device.metadata}")


async def scan_for_devices():
    global latest_readings
    scanner = BleakScanner(detection_callback)

    await scanner.start()
    print("Scanning for devices...")

    await asyncio.sleep(30)  # Scan for devices for 10 seconds
    await scanner.stop()

    if len(latest_readings) < 2:
        print("Could not find two devices.")
        return False

    print("Found devices:", latest_readings)
    return True

def write_to_csv(elapsed_time, outside_temp, outside_humidity, inside_temp, inside_hum, filename='data.csv'):
    # Open the file in append mode 'a' to add data without overwriting
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file is empty, write the header first
        if file.tell() == 0:
            writer.writerow(['elapsed_time', 'outside_temp', 'outside_humidity', 'inside_temp'])
        
        # Write the new row of data
        writer.writerow([elapsed_time, outside_temp, outside_humidity, inside_temp, inside_hum])

async def log_readings():
    global time_array, T_inside, T_outside, H_inside, H_outside, ml_model

    while True:
        await scan_for_devices()  # Re-scan devices to fetch latest readings
        await asyncio.sleep(60)  # Wait for 1 minute (60 seconds)

        # Calculate time since the start of the program (in minutes)
        elapsed_time = int((time.time() - start_time) / 60)

        # Format the readings
        outside_temp_hum = latest_readings.get("GVH5075_4732")  # OutsideTemp (Device 1)
        inside_temp_hum = latest_readings.get("GVH5075_82A7")  # InsideTemp (Device 2)

        # Get temperatures and humidity (in °F and %)
        outside_temp, outside_hum = outside_temp_hum if outside_temp_hum else (None, None)
        inside_temp, inside_hum = inside_temp_hum if inside_temp_hum else (None, None)

        # Append data to numpy arrays if readings are available
        if inside_temp is not None and outside_temp is not None:
            time_array = np.append(time_array, elapsed_time)
            T_inside = np.append(T_inside, inside_temp)
            T_outside = np.append(T_outside, outside_temp)
            H_inside = np.append(H_inside, inside_hum)
            H_outside = np.append(H_outside, outside_hum)

            # Write to CSV file as well
            write_to_csv( elapsed_time, outside_temp, outside_hum, inside_temp, inside_hum )

            # Log information to the GUI
            #log_message = f"Time: {elapsed_time} min, Inside Temp: {inside_temp:.2f}°F, Outside Temp: {outside_temp:.2f}°F\n"
            log_message = f"Time: {elapsed_time:>5} min, Inside Temp: {inside_temp:>8.2f}°F, Outside Temp: {outside_temp:>8.2f}°F\n"

            update_log(log_message)

            future_conditions = np.array([[elapsed_time, outside_temp, outside_hum]])

            # Train ML model if we have enough data
            if len(time_array) > 8:  # Adjust threshold as needed
                if ml_model is None:

                    def read_csv_to_numpy_arrays(file_path):
                       df = pd.read_csv(file_path)
                       numpy_arrays = [df[column].to_numpy() for column in df.columns]
                       return numpy_arrays

                    training_data = read_csv_to_numpy_arrays( 'your_historical_data.csv' )
                    # Prepare the dataset
                    #X = np.column_stack((time_array, T_outside, H_inside, H_outside))
                    # time_array, outside_temp,outside_humidity
                    X = np.column_stack((training_data[0], training_data[1], training_data[2]))
                    y = training_data[3] 
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train Random Forest model
                    ml_model = RandomForestRegressor()
                    ml_model.fit(X_train, y_train)

                    # Evaluate the model
                    y_pred = ml_model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    #update_log(f"Mean Squared Error of ML Model: {mse:.4f}\n")

                # Make predictions
                predicted_temp_ml = ml_model.predict(future_conditions)

                # Proceed only if we have enough data points for cooling model optimization
                if len(time_array) > 8:
                    k_optimal = optimize_cooling_model()  # Get optimal k
                    predicted_temp_cooling = inside_temp - k_optimal * (inside_temp - outside_temp) * 1  # 1 minute prediction

                    # Combine predictions
                    combined_prediction = (predicted_temp_ml[0] + predicted_temp_cooling) / 2

                    # Update predictions in GUI
                    update_predictions(predicted_temp_ml[0], predicted_temp_cooling, combined_prediction)

                    # Additional predictions for 5, 10, 20, 30, and 45 minutes
                    future_timeframes = [5, 10, 20, 30, 45]  # in minutes
                    additional_predictions = []
                    for future_time in future_timeframes:
                        predicted_temp_ml_future = ml_model.predict(np.array([[elapsed_time + future_time, outside_temp, outside_hum]]))
                        predicted_temp_cooling_future = inside_temp - k_optimal * (inside_temp - outside_temp) * future_time
                        combined_future = (predicted_temp_ml_future[0] + predicted_temp_cooling_future) / 2
                        additional_predictions.append((future_time, predicted_temp_ml_future[0], predicted_temp_cooling_future, combined_future))

                    # Update future predictions in GUI
                    update_future_predictions(additional_predictions)

                    global desired_temp
                    target_temp = desired_temp 
                    #time_to_reach_target = (inside_temp - target_temp)/(k_optimal * (inside_temp - outside_temp))
                    print( f'k_optimal: {k_optimal} desired_temp: {desired_temp}, outside_temp: {outside_temp}, inside_temp: {inside_temp}' )
                    time_to_reach_target = (-1/k_optimal)*math.log(( desired_temp - outside_temp)/(inside_temp-outside_temp ))

                    global inprogress 
                    if ( desired_temp < inside_temp ) and not inprogress:
                       switch_image( 'open' )
                       switch_curtain_image( 'open' )
                       switch_fan_image( 'open' )
                       target_temp_label.config(text=f"Target : {target_temp}°F\nStarting Exhaust/Gable fans...\nSwitchbot opening Curtains...\nOpen Windows now!!!\n\nTarget temperature will be reached in {time_to_reach_target:.2f} minutes")

                       def show_alert():
                          messagebox.showinfo("Alert", "Open Windows now!!!")
                       show_alert()
                       inprogress = True

                    if inprogress and ( desired_temp >= inside_temp ):
                       switch_image( 'closed' )
                       switch_curtain_image( 'closed' )
                       switch_fan_image( 'closed' )
                       target_temp_label.config(text=f"Target : {target_temp}°F\nStopping Exhaust/Gable fans...\nSwitchbot closing Curtains...\nClose Windows now!!!\n\nTarget temperature reached")

                       def show_alert():
                          messagebox.showinfo("Alert", "Close Windows now!!!")
                       show_alert()
                       inprogress = False
                     

                #else:
                #    update_log("Not enough data points for cooling model optimization.\n")
            # else:
            #    update_log("Not enough data points for ML model training.\n")


def optimize_cooling_model():
    global time_array, T_inside, T_outside

    if len(time_array) < 2:
        print("Not enough data points for cooling model optimization.")
        return None

    # Time step
    dt = time_array[1] - time_array[0]
    dt = 1

    # Define the cooling model
    def model(k, T_inside, T_outside, dt):
        T_model = np.zeros_like(T_inside)
        T_model[0] = T_inside[0]  # Initial inside temperature

        # Euler method to simulate temperature over time
        for i in range(1, len(time_array)):
            T_model[i] = T_model[i - 1] - k * (T_model[i - 1] - T_outside[i - 1]) * dt
        return T_model

    # Define the objective function (MSE between actual and model-predicted inside temperature)
    def objective(k, T_inside, T_outside, dt):
        T_model = model(k, T_inside, T_outside, dt)
        return np.mean((T_inside - T_model) ** 2)

    # Optimize k value
    k_initial = 0.1  # Initial guess
    result = minimize(objective, k_initial, args=(T_inside, T_outside, dt), bounds=[(0, None)])

    return result.x[0] if result.success else None

# Define the device info label once, outside of the function
fixed_width_font = font.Font(family="Courier New", size=16)
device_label = tk.Label(device_frame, text="", anchor="w", font=fixed_width_font)
device_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")


def update_device_info():
    """Update the device information in the GUI by clearing the old values first."""
    # Generate the new device information string

    device_info = "\n".join([
    f"{('Inside Sensor' if device == 'GVH5075_82A7' else 'Outside Sensor'): <15} [{' ' + device:<15}]: Temp = {temp:>6.2f}°F, Hum = {hum:>6.2f}%"
    for device, (temp, hum) in latest_readings.items()
    ])

    # Update the text of the existing label with new information
    device_label.config(text=device_info)

def update_log(message):
    """Update the log in the GUI."""
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, message)
    log_text.config(state=tk.DISABLED)
    log_text.see(tk.END)  # Auto-scroll to the latest log

def update_predictions(pred_ml, pred_cooling, combined):
    """Update the predictions in the GUI."""
    prediction_label.config(text=f"ML Prediction: {pred_ml:.2f}°F, Cooling Model Prediction: {pred_cooling:.2f}°F, Combined: {combined:.2f}°F")

def update_future_predictions(predictions):
    """Update future predictions in the GUI."""
    future_info = "\n".join([f"In {t:3} min: ML Model = {ml:.2f}°F, Cooling Model = {cooling:.2f}°F, Combined = {combined:.2f}°F"
                              for t, ml, cooling, combined in predictions])
    future_predictions_label.config(text=f"Future Predictions:\n{future_info}")

def start_logging_thread():
    """Run the logging process in a separate thread."""
    asyncio.run(log_readings())

def start_async_tasks():
    """Run the async scanning and logging process."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(scan_for_devices())

# Start scanning and logging in separate threads
threading.Thread(target=start_async_tasks, daemon=True).start()
threading.Thread(target=start_logging_thread, daemon=True).start()

# Start the tkinter main loop
root.mainloop()

