# Parkinson's Disease Detection 

A machine learning-based application for early detection of Parkinson's Disease through voice analysis using MFCC (Mel-frequency cepstral coefficients) features.

## Overview

This tool uses audio processing and machine learning techniques to analyze voice recordings and detect potential indicators of Parkinson's Disease. The system implements multiple machine learning models including Random Forest, SVM, KNN, and CNN for robust prediction.

## Features

- **Multiple Model Support**: Implements four different machine learning models:
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Convolutional Neural Network (CNN)

- **Interactive GUI**:
  - Upload pre-recorded audio files
  - Record live audio for analysis
  - Real-time classification
  - Visualization of prediction probabilities
  - Model accuracy plots

- **Audio Processing**:
  - MFCC feature extraction
  - Real-time audio recording capability
  - Support for WAV file format

## Requirements

```
python >= 3.12
tensorflow
librosa
sounddevice
scipy
numpy
pandas
scikit-learn
pillow
matplotlib
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/parkinsons-detection-tool.git
cd parkinsons-detection-tool
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Usage

### GUI Interface

1. **Launch the Application**:
   - Run `app.py` to start the application
   - Navigate through the welcome screen to access the main tool

2. **Audio Analysis**:
   - Upload an existing audio file (.wav format)
   - Record live audio (customizable duration)
   - View real-time predictions and probability scores

3. **Visualization**:
   - View likelihood plots for predictions
   - Access model accuracy metrics
   - Analyze prediction confidence scores

### Notebook Interface

The `nist.ipynb` notebook provides:
- Model training pipeline
- Feature extraction demonstration
- Performance metrics visualization
- Interactive audio classification


## Project Structure

```
├── app.py                 # Main application file
├── nist.ipynb            # Training notebook
├── requirements.txt      # Project dependencies
├── models/
│   ├── cnn_parkinsons_model.h5
│   ├── rfc_trained_model.joblib
│   ├── svm_trained_model.joblib
│   └── knn_trained_model.joblib
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sourced from approved medical datasets
- Built using TensorFlow and scikit-learn frameworks
- Uses MFCC feature extraction techniques from librosa

## Contact

For questions and support, please open an issue in the GitHub repository.

## Note

This tool is intended for research and screening purposes only and should not be used as a definitive diagnostic tool. Always consult healthcare professionals for medical advice.
