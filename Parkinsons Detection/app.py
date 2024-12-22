from tkinter import Tk, Label, Button, Canvas, Frame, StringVar, Entry, filedialog
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
from tensorflow.keras import models


# Function to extract MFCC features
def feature_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type="kaiser_fast")
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=22).T, axis=0)
    return mfcc


# Load pre-trained CNN model
cnn_model = models.load_model('cnn_parkinsons_model.h5')


# Function to open the main Parkinson's detection tool window
def open_detection_tool():
    # Create a new Tkinter window for the detection tool
    detection_tool_window = Tk()
    detection_tool_window.title("Parkinson's Disease Detection")
    detection_tool_window.geometry("800x800")

    # Load the background image
    background_image_path = r"/Users/tusharshetty/Desktop/invocation/fish.png"  # Replace with your image path
    original_image = Image.open(background_image_path)
    background_image = ImageTk.PhotoImage(original_image.resize((800, 800), Image.Resampling.LANCZOS))

    # Create the main frame
    main_frame = Frame(detection_tool_window, bg="#eaf7ff")
    main_frame.pack(fill="both", expand=True)

    canvas_main = Canvas(main_frame, width=800, height=800)
    canvas_main.pack(fill="both", expand=True)
    canvas_main.create_image(0, 0, anchor="nw", image=background_image)

    # Frame to place widgets in the foreground
    frame = Frame(main_frame, bg="#eaf7ff", bd=2, relief="solid", width=750, height=700)
    frame.place(relx=0.5, rely=0.5, anchor="center")

    # Variables for dynamic updates
    file_path_var = StringVar(value="No file selected")
    duration_var = StringVar(value="5")
    predictions_var = StringVar(value="Predictions will appear here.")
    likelihood_var = StringVar(value="Likelihood: Not calculated yet.")

    # Function to classify audio files
    def classify_audio(file_path):
        if not os.path.exists(file_path):
            predictions_var.set("Error: File not found.")
            return

        mfcc_features = feature_extraction(file_path).reshape(1, -1)

        # CNN Prediction (get probabilities for likelihood)
        cnn_probabilities = cnn_model.predict(mfcc_features.reshape(1, -1, 1))[0][0]
        cnn_prediction = (cnn_probabilities > 0.5).astype("int32")

        # Update predictions in the GUI
        final_prediction = "PwPD (Parkinson's Detected)" if cnn_prediction == 1 else "HC (Healthy Control)"
        predictions_var.set(f"Final Prediction: {final_prediction}")

        # Update likelihood
        likelihood_var.set(f"Likelihood of Parkinson's: {cnn_probabilities * 100:.2f}%")

    # Function to select and classify an audio file
    def upload_and_classify():
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        file_path_var.set(file_path if file_path else "No file selected")
        if file_path:
            classify_audio(file_path)

    # Function to record live audio and classify it
    def record_and_classify():
        try:
            duration = int(duration_var.get())
        except ValueError:
            predictions_var.set("Error: Invalid duration.")
            return

        sample_rate = 22050
        predictions_var.set("Recording...")
        detection_tool_window.update()

        # Record audio
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        write(temp_file.name, sample_rate, audio)

        # Classify the recorded audio
        classify_audio(temp_file.name)
        os.unlink(temp_file.name)

    # Function to plot the likelihood of Parkinson's
    def plot_likelihood():
        likelihood = float(likelihood_var.get().split(": ")[1].replace("%", ""))

        plt.figure(figsize=(6, 4))
        plt.bar(["Healthy Control", "Parkinson's Detected"], [100 - likelihood, likelihood], color=["green", "red"])
        plt.title("Likelihood of Parkinson's Disease", fontsize=16)
        plt.ylabel("Probability (%)", fontsize=14)
        plt.xticks(fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Display the graph
        plt.show()

    # Function to plot model accuracy (simulated data for illustration)
    def plot_model_accuracy():
        # Simulated accuracy data (replace with your real training history data)
        epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        training_accuracy = [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.96, 0.97]
        validation_accuracy = [0.83, 0.86, 0.88, 0.89, 0.91, 0.91, 0.92, 0.93, 0.94, 0.94]

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, training_accuracy, label="Training Accuracy", marker="o", color="blue")
        plt.plot(epochs, validation_accuracy, label="Validation Accuracy", marker="o", color="orange")
        plt.title("Model Accuracy Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xticks(epochs)
        plt.ylim(0.8, 1.0)
        plt.legend()
        plt.grid(True)

        # Display the graph
        plt.show()

    # Widgets for Upload, Record, Plotting, and Predictions
    Label(
        frame, text="Upload an Audio File:", font=("Times New Roman", 16, "bold"), bg="#eaf7ff"
    ).pack(pady=10)
    Button(
        frame, text="Upload and Classify", command=upload_and_classify, font=("Times New Roman", 14),
        bg="#4caf50", fg="white", relief="raised", bd=3, padx=10, pady=5
    ).pack(pady=10)

    Label(
        frame, text="Record Live Audio:", font=("Times New Roman", 16, "bold"), bg="#eaf7ff"
    ).pack(pady=10)
    Label(
        frame, text="Duration (seconds):", font=("Times New Roman", 14), bg="#eaf7ff"
    ).pack(pady=5)
    Entry(
        frame, textvariable=duration_var, width=10, font=("Times New Roman", 14), justify="center"
    ).pack(pady=5)

    Button(
        frame, text="Record and Classify", command=record_and_classify, font=("Times New Roman", 14),
        bg="#2196f3", fg="white", relief="raised", bd=3, padx=10, pady=5
    ).pack(pady=10)

    Button(
        frame, text="Plot Likelihood", command=plot_likelihood, font=("Times New Roman", 14),
        bg="#ff9800", fg="white", relief="raised", bd=3, padx=10, pady=5
    ).pack(pady=10)

    Button(
        frame, text="Plot Model Accuracy", command=plot_model_accuracy, font=("Times New Roman", 14),
        bg="#795548", fg="white", relief="raised", bd=3, padx=10, pady=5
    ).pack(pady=10)

    Label(
        frame, text="Predictions:", font=("Times New Roman", 16, "bold"), bg="#eaf7ff"
    ).pack(pady=10)
    Label(
        frame, textvariable=predictions_var, wraplength=350, justify="left", bg="#eaf7ff",
        font=("Times New Roman", 14)
    ).pack(pady=10)

    Label(
        frame, textvariable=likelihood_var, wraplength=350, justify="left", bg="#eaf7ff",
        font=("Times New Roman", 14), fg="blue"
    ).pack(pady=10)

    # Start the new window's event loop
    detection_tool_window.mainloop()


# Introductory window
def intro_window():
    intro = Tk()
    intro.title("Welcome to Parkinson Detection Tool")
    intro.geometry("800x800")

    # Load the background image
    bg_image_path = r"/Users/tusharshetty/Desktop/invocation/fis.png"  # Replace with the correct image path
    bg_image = Image.open(bg_image_path)
    bg_photo = ImageTk.PhotoImage(bg_image.resize((800, 800), Image.Resampling.LANCZOS))

    canvas = Canvas(intro, width=800, height=800)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=bg_photo)

    # Overlay with adjusted placement
    overlay = Frame(intro, bg="#ffffff", width=500, height=200, relief="solid", bd=2)
    overlay.place(relx=0.5, rely=0.75, anchor="center")  # Position below the logo

    Label(
        overlay, text="Welcome to Parkinson's Detection Tool", font=("Times New Roman", 20, "bold"),
        bg="#ffffff", wraplength=450, justify="center"
    ).pack(pady=20)

    Button(
        overlay, text="Proceed to Detection Tool", font=("Times New Roman", 16, "bold"),
        bg="#4caf50", fg="white", command=lambda: [intro.destroy(), open_detection_tool()]
    ).pack(pady=20)

    intro.mainloop()


# Run the intro window
intro_window()