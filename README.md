# Customer Churn Prediction using Artificial Neural Network

This project uses a deep learning model to predict customer churn in a banking context. The project includes model training, experimentation, and a web-based Streamlit application for making predictions.

**Live Demo:** https://ann-classification-churn-6cqx4prcvp6zbtbh4gnhmp.streamlit.app/

## Project Overview

**Objective:** Predict whether a customer is likely to leave (churn) a bank based on their profile and behavior.

**Approach:** Build and train an Artificial Neural Network (ANN) using TensorFlow/Keras with customer data.

## Project Structure

```
dlprojectANN/
├── Churn_Modelling.csv        # Dataset with customer information
├── experiments.ipynb            # Model training and experimentation notebook
├── prediction.ipynb             # Prediction notebook for testing
├── app.py                       # Streamlit web application
├── model.h5                     # Trained model (HDF5 format)
├── requirements.txt             # Python dependencies
├── scaler.pkl                   # Saved StandardScaler for feature scaling
├── label_encoder_gender.pkl     # Saved encoder for gender column
├── onehot_encoder_geo.pkl       # Saved encoder for geography column
└── logs/                        # TensorBoard training logs
    └── fit/
```

## Dataset

**File:** `Churn_Modelling.csv`

The dataset contains customer information with the following key features:
- **CreditScore**: Customer's credit score
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: How long the customer has been with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of products used
- **HasCrCard**: Whether customer has a credit card
- **IsActiveMember**: Activity status
- **EstimatedSalary**: Estimated annual salary
- **Exited**: Target variable (1 = churned, 0 = retained)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd dlprojectANN
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

Run the training notebook to build and train the ANN:
```bash
jupyter notebook experiments.ipynb
```

This notebook:
- Loads and preprocesses the data
- Encodes categorical variables (Gender, Geography)
- Scales numerical features
- Builds a 3-layer neural network
- Trains the model with early stopping and TensorBoard monitoring
- Saves the trained model and encoders

### 2. Making Predictions

Use the Streamlit web application for interactive predictions:
```bash
streamlit run app.py
```

The app provides an interactive interface where you can:
- Input customer information via dropdown menus and sliders
- Get real-time churn probability prediction
- See whether the customer is likely to churn or not

### 3. Experimentation & Analysis

Explore predictions and model behavior in the prediction notebook:
```bash
jupyter notebook prediction.ipynb
```

## Model Architecture

The ANN consists of 3 layers:

```
Input Layer (11 features)
    ↓
Dense Layer 1: 64 neurons + ReLU activation
    ↓
Dense Layer 2: 32 neurons + ReLU activation
    ↓
Output Layer: 1 neuron + Sigmoid activation
```

**Configuration:**
- **Optimizer:** Adam (learning_rate=0.01)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy
- **Early Stopping:** Monitor validation loss with patience=5

## Key Files Explained

| File | Purpose |
|------|---------|
| `experiments.ipynb` | Data preprocessing, model building, and training |
| `prediction.ipynb` | Testing and validation notebook |
| `app.py` | Interactive Streamlit web application |
| `model.h5` | Trained neural network (saved in HDF5 format) |
| `scaler.pkl` | StandardScaler for normalizing features |
| `label_encoder_gender.pkl` | LabelEncoder for gender variable |
| `onehot_encoder_geo.pkl` | OneHotEncoder for geography variable |

## Data Preprocessing Steps

1. **Drop unnecessary columns:** RowNumber, CustomerId, Surname
2. **Encode Gender:** LabelEncoder (0 = Female, 1 = Male)
3. **Encode Geography:** OneHotEncoder (France, Germany, Spain)
4. **Feature Scaling:** StandardScaler for all features
5. **Train-Test Split:** 80% training, 20% testing

## Technologies Used

- **TensorFlow/Keras:** Deep learning framework
- **Pandas:** Data manipulation
- **Scikit-learn:** Preprocessing and encoding
- **Streamlit:** Web application framework
- **TensorBoard:** Training visualization
- **NumPy:** Numerical operations

## Training Monitoring

TensorBoard logs are saved during training:
```bash
tensorboard --logdir logs/fit
```

View training progress including loss and accuracy metrics.

## Model Performance

The model is trained to minimize binary crossentropy loss with early stopping to prevent overfitting. Training history is logged and can be visualized via TensorBoard.

## Dependencies

See `requirements.txt` for the complete list:
- tensorflow-cpu==2.20.0
- pandas
- numpy
- scikit-learn
- tensorboard
- matplotlib
- streamlit

## How to Make Predictions

**Using the Streamlit App (Recommended):**
1. Run `streamlit run app.py`
2. Input customer details in the interface
3. View churn probability and prediction

**Programmatically:**
```python
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Load model and encoders
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input data (must match training preprocessing)
# Scale and predict
prediction = model.predict(input_scaled)
churn_probability = prediction[0][0]
```

## Notes

- All input data is preprocessed and scaled using the same encoders and scaler used during training
- The model outputs a probability between 0 and 1 (sigmoid activation)
- Threshold for churn prediction is 0.5 (>0.5 = likely to churn, <0.5 = unlikely to churn)

## Author Notes

This project demonstrates a complete ML pipeline:
- Data preprocessing and feature engineering
- Neural network design and training
- Model persistence and deployment
- User-friendly prediction interface

---

For questions or improvements, refer to the notebooks for detailed implementation details.
