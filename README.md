# 🧠 Deep Learning Projects: Churn Classification & Salary Prediction with ANN

This repository contains two complete machine learning projects built with **Artificial Neural Networks (ANN)**:

1. 🔁 **Customer Churn Classification** – Predict whether a customer will leave a service.
2. 💼 **Salary Prediction Regression** – Predict a person's salary based on input features.

Both models are implemented using **TensorFlow/Keras** and deployed via **Streamlit** with proper preprocessing, encoding, scaling, and model saving using `pickle`.

---

## 📌 Projects Overview

### 1. 🔁 Churn Classification

- **Type:** Binary Classification
- **Goal:** Predict if a customer will churn
- **Techniques:**
  - ANN with binary cross-entropy loss
  - Label & OneHot Encoding
  - Feature scaling
  - Hyperparameter tuning using GridSearch
  - Final model deployment with Streamlit UI

### 2. 💼 Salary Prediction

- **Type:** Regression
- **Goal:** Predict salary based on demographic and geographic features
- **Techniques:**
  - ANN with MSE loss
  - Label encoding for gender
  - OneHot encoding for location
  - Model saved and deployed with Streamlit

---

## 🗂️ Project Structure

📦 root/
│
├── app.py # Streamlit app for one or both models
├── experiments.ipynb # General experiments and model testing
├── prediction.ipynb # Churn prediction notebook
├── salaryregression.ipynb # Salary prediction model training
├── hyperAnn.ipynb # Base ANN architecture
├── hyperparametertuningann.ipynb # Grid search tuning
│
├── model.h5 # Churn model (saved ANN)
├── salary_regression_model.h5 # Salary prediction model
│
├── *.pkl # All encoders and scalers (gender, geo, etc.)
│
├── requirements.txt # Dependencies list
└── README.md # Project documentation

yaml
Copy
Edit

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/deep-learning-ann-projects.git
cd deep-learning-ann-projects
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🚀 Run the Streamlit App
To launch the web app locally:

bash
Copy
Edit
streamlit run app.py
Fill in the form inputs and receive a prediction (either for churn classification or salary prediction depending on how the app is configured).

⚙️ Technologies Used
Python 3.x

TensorFlow / Keras

Streamlit

Pandas, NumPy

scikit-learn

Pickle

📈 Model Summary
Task	Model Type	Loss Function	Accuracy / Metric
Churn Classification	ANN	Binary Crossentropy	Accuracy, Precision, Recall
Salary Prediction	ANN	MSE	MAE, R² Score

🧪 Future Work
Add support for more models (e.g., decision trees, XGBoost)

Add charts/metrics to Streamlit app

Export results as CSV

Add unit testing

📜 License
This project is open-source and available under the MIT License.

🙌 Acknowledgments
Thanks to:

Streamlit

TensorFlow

scikit-learn

Open datasets for providing training material

vbnet
Copy
Edit

### ✅ Next Step

Let me know:
- What dataset you used (so I can add links/credits)
- Do you want **separate Streamlit apps** or a **single UI** with a dropdown to switch between tasks?
- Would you like me to help you write a `requirements.txt` file too?

Once you confirm, I’ll polish it further for publishing.
