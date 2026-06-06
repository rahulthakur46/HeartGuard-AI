# 🫀 Heart Disease Prediction ML App

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.22.0-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> A clinical-grade, interactive heart disease risk prediction web application built with Machine Learning and deployed on Streamlit Cloud.

**🔗 Live Demo:** [Click Here to Open App](https://your-app-link.streamlit.app)  
**👤 Author:** Rahul Thakur  
**📅 Year:** 2026

---

## 📸 App Preview

```
🫀 HeartGuard AI
├── 🔍 Predict & Explain     → Real-time risk prediction + feature contribution
├── 📊 Data Explorer         → Filter, visualize, download dataset
├── 📈 Visual Analytics      → 3D plots, scatter matrix, heatmaps
├── 🧠 Model Report          → ROC curve, confusion matrix, cross-validation
├── 💡 Health Guide          → Clinical risk factors & normal ranges
└── ℹ️  About                → Tech stack, model specs, metrics
```

---

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| ✅ Accuracy | **81.97%** |
| 🎯 Precision | **88.89%** |
| 📢 Recall | **75.00%** |
| ⚖️ F1 Score | **81.36%** |
| 📈 ROC-AUC | **86.37%** |
| 🔁 CV Mean (5-Fold) | **75.22% ± 5.44%** |

---

## 🚀 Features

- 🔬 **Real-time Prediction** — Instant heart disease risk assessment from 13 clinical inputs
- 📊 **Feature Contribution** — LIME-style explainability showing which features drove the prediction
- 🌐 **Interactive 3D Visualizations** — Rotate and explore feature space in 3D
- 🧩 **Scatter Matrix** — Multi-feature pairwise comparison colored by diagnosis
- 🫧 **Bubble Chart** — Age vs Blood Pressure vs Cholesterol in one view
- 🌡️ **Correlation Heatmap** — Full feature correlation matrix
- 🔎 **Live Data Filtering** — Query dataset using pandas syntax (`age > 55`)
- ⬇️ **CSV Report Download** — Export patient summary and filtered data
- 📉 **ROC Curve & AUC** — Professional model evaluation
- 🔁 **5-Fold Cross Validation** — Visual fold-by-fold accuracy breakdown
- 🌳 **Decision Tree Rules** — Human-readable if-else logic export
- 📂 **Custom CSV Upload** — Analyze your own dataset on the fly
- 💡 **Health Guide** — Clinical normal ranges and heart-healthy habits

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly (interactive) |
| Model Persistence | Joblib |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
Heart-Disease-Prediction-ML/
│
├── heart_disease_app.py       # Main Streamlit application
├── heart_disease_model.pkl    # Trained Decision Tree model
├── heart.csv                  # UCI Heart Disease dataset
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/RahulThakur/Heart-Disease-Prediction-ML.git
cd Heart-Disease-Prediction-ML
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run heart_disease_app.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 📊 Dataset Info

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Patients | 303 |
| Features | 13 clinical attributes |
| Target | Binary (0 = Healthy, 1 = Heart Disease) |
| Disease Prevalence | 54.5% |
| Missing Values | None |

### Features Description

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numeric |
| `sex` | Sex (1=Male, 0=Female) | Binary |
| `cp` | Chest pain type (0–3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numeric |
| `chol` | Serum cholesterol (mg/dl) | Numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG result (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numeric |
| `exang` | Exercise induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | Numeric |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0–3) | Numeric |
| `thal` | Thalassemia type (0–3) | Categorical |

---

## 🧠 Model Details

```
Algorithm     : Decision Tree Classifier
Criterion     : Entropy (Information Gain)
Max Depth     : 6
Min Samples Split : 4
Min Samples Leaf  : 4
Test Split    : 80% Train / 20% Test
Random State  : 42
```

---

## 🖥️ App Screenshots

> *(Add screenshots here after deployment)*
>
> Tip: Press `F12` in browser → Screenshot, or use Windows `Win + Shift + S`

---

## ⚠️ Disclaimer

This application is built **strictly for educational and portfolio purposes**.  
It is **not a certified medical device** and should **never** replace professional medical advice, diagnosis, or treatment. Always consult a qualified cardiologist for clinical decisions.

---

## 📬 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rahul_Thakur-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-RahulThakur-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RahulThakur)
[![Email](https://img.shields.io/badge/Email-yourmail@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:yourmail@gmail.com)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">
  Made with ❤️ by <b>Rahul Thakur</b>
</div>