# 🏏 IPL Score Prediction using Deep Learning

Cricket is no longer just about instincts — it's about **data-driven decisions**. In high-stakes matches like the Indian Premier League (IPL), every run, every over, and every wicket can shift the momentum. This project uses deep learning to predict the final score of a batting team in a T20 match, based on real-time inputs like overs, runs, wickets, and team details.

---

## ❓ Why Use Deep Learning for Score Prediction?

While traditional ML models struggle with complex data, deep learning excels:

- 📈 **Pattern Recognition**: Learns from historical IPL datasets automatically  
- ⚡ **Multivariate Analysis**: Handles many inputs like teams, overs, venue, wickets  
- 🎯 **High Accuracy**: Outperforms simpler models by capturing non-linear relationships  
- 🧠 **Real-Time Use**: Ideal for broadcasters, analysts, apps, and fantasy platforms

---

## ✨ Project Highlights

- ✅ Deep Neural Network (DNN) built using **TensorFlow (Keras)**
- 🧼 Preprocessing: Label Encoding, Feature Scaling, Feature Selection
- 📊 Visualizations using **Matplotlib** and **Seaborn**
- 🧪 Evaluated with **Mean Absolute Error (MAE)**
- 🖱️ Interactive prediction using **ipywidgets**
- 📁 Historical data: IPL seasons **2008 to 2017**

---

## 📸 Screenshots

### 🎯 Prediction Output

> A live prediction of the final score based on match conditions:

![Prediction Screenshot](output-images/model_output.png)

### 📉 Model Training (Optional)

> MAE and loss over training epochs:

![Training Loss](output-images/2.png)

---

## 🔧 Project Pipeline

### 📦 Installing Libraries

Import required libraries such as:

- `numpy`, `pandas` – Data processing  
- `matplotlib`, `seaborn` – Visualizations  
- `scikit-learn` – Preprocessing & model tools  
- `tensorflow.keras` – Deep learning model  
- `ipywidgets` – Interactive input for predictions

### 📂 Loading the Dataset

- Load IPL dataset from 2008–2017  
- Contains batting/bowling teams, venue, players, overs, wickets, etc.  
- Load data with `pandas.read_csv()` into DataFrame

### 🔤 Encoding

- Apply **Label Encoding** to convert text into numbers  
- Store encoders in a dictionary for consistent transformation  
- Ensure all categorical columns are transformed correctly

### 🏋️‍♂️ Training

- Features selected and split using `train_test_split()`  
- Scale features using `MinMaxScaler`  
- Neural network architecture: `Sequential`, `Dense`, `ReLU`, `Linear`  
- Compiled with **Huber Loss** and **Adam optimizer**

### 📈 Evaluation

- Evaluate model on test set using **Mean Absolute Error (MAE)**  
- Predict and compare actual vs predicted scores

### 🖱️ Interactive Widget

- Built using `ipywidgets.Dropdown`, `Button`, `display()`  
- Users select real-time match conditions to predict final score  
- Widget automatically encodes inputs, applies scaler, and predicts

---

## 🚀 How to Run This Project

To run this project locally:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rohitlakha/ipl-score-prediction.git
cd ipl-score-prediction
