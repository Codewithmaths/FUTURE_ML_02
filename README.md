# 📉 Customer Churn Prediction System – FUTURE_ML_02

Welcome to the **Customer Churn Prediction System**, a machine learning project built to identify customers who are likely to stop using a service. This tool is especially valuable for telecom, SaaS, and banking sectors, where retaining existing customers is more cost-effective than acquiring new ones.

## 🚀 Project Overview

This hands-on machine learning system:
- Predicts the probability that a customer will churn.
- Highlights the most important features influencing churn.
- Provides both a report and a web interface for easy interaction.

📁 Repository Structure

FUTURE_ML_02/
├── UI.py # Streamlit web app for uploading and predicting <br>
├── model_training.ipynb # Jupyter notebook for training and evaluation<br>
├── xgboost_model.pkl # Trained XGBoost model<br>
├── churn_predictions.csv # Example prediction output<br>
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dataset used (Telco churn)<br>
├── requirements.txt # Required Python libraries<br>
└── README.md # Project documentation<br>


---

## 🧠 Features

✔️ Churn probability for each customer  
✔️ Feature importance chart (driving factors)  
✔️ Confusion matrix, ROC-AUC, Precision-Recall  
✔️ Interactive UI with Streamlit  
✔️ Business-ready PDF report or downloadable CSV  

---

## 🛠️ Tools & Libraries

- **Python**
- **Scikit-learn**
- **XGBoost**
- **Pandas & NumPy**
- **Matplotlib/Seaborn**
- **Streamlit** (for the user interface)

---

## 📊 Dataset

We use the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which includes customer demographics, account details, and service usage patterns.

---

## ⚙️ How to Use

1. 🔧 Install Dependencies

```bash
pip install -r requirements.txt

2. 🧠 Train the Model (Optional)
Open and run all cells in model_training.ipynb to clean data, train the model, and generate evaluation visuals.

3. 💻 Run Streamlit App
streamlit run UI.py

Upload a CSV file with valid customer data to get churn predictions instantly.

📂 Sample Input Format
Make sure your CSV has the following columns:

bash
Copy
Edit
['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
 'MonthlyCharges', 'TotalCharges']
📌 Example
Try uploading a cleaned CSV and get:

Churn probability per customer

Prediction result (Yes/No)

Downloadable CSV of results

📄 License
This project is open-source under the MIT License.

🤝 Contributing
Feel free to fork the repo and submit pull requests. Feedback and collaboration are welcome!

👤 Author
Manav Singh
🌐 GitHub: Codewithmaths

⭐ Star the repo if you found it helpful!
yaml
Copy
Edit

