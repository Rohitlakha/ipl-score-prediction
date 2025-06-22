# 🏏 IPL Score Prediction Using Deep Learning

This project predicts the final score of an IPL (Indian Premier League) team during a T20 match using a deep learning model. It uses match features like venue, teams, wickets, overs, and more to generate accurate predictions. The model is trained on past IPL data and provides an interactive way to test predictions using ipywidgets.

---

---
## 📷 Prediction Screenshot

Here is an example of the model predicting the final IPL score based on match inputs:

![IPL Score Prediction Output](output-images/model_output.png)

## 1. 📦 Installing Libraries

We are importing all necessary Python libraries such as NumPy, Pandas, Scikit-learn, Matplotlib, Keras and Seaborn required for data handling, visualization, preprocessing and building deep learning models.

---

## 2. 📂 Loading the Dataset

The dataset can be downloaded from here. It contains data from 2008 to 2017 and contains features like venue, date, batting and bowling team, names of batsman and bowler, wickets and more. We will load the IPL cricket data from CSV files into pandas DataFrames to explore and prepare for modeling.

---

## 3. 📊 Exploratory Data Analysis

We will do Exploratory Data Analysis (EDA) to analyze how many unique matches have been played at each venue by counting distinct match IDs for every venue. Then, we’ll visualize this data using a horizontal bar chart to see which venues host the most matches.

---

## 4. 🔤 Performing Label Encoding

We will convert categorical text data into numeric labels using Label Encoding because ML models work with numbers.

- `LabelEncoder()` converts text labels into integers.  
- `fit_transform()` learns encoding and applies it.  
- `copy()` : creates a duplicate of the DataFrame to avoid changing the original data  
- A dictionary assignment stores each encoder for future use like decoding or consistent transformation

---

## 5. 🧾 Performing Feature Selection

We drop `date` and `mid` columns because they are identifiers and don’t provide meaningful information for correlation analysis. By removing these irrelevant columns we focus on features that can reveal relationships useful for modeling or insights.

- `drop()` : removes specified columns from the DataFrame  
- `corr()` : computes pairwise correlations between numerical features  
- `sns.heatmap()` : creates a colored matrix to visualize correlations with values  
- `plt.show()` : displays the plot on screen

---

## 6. ✂️ Splitting the Dataset into Training and Testing

We will select relevant features and the target variable then split the data into training and testing sets for model building and evaluation.

- `data_encoded[feature_cols]` : selects specified columns as features  
- `train_test_split()` : splits features and target into training and test subsets  
- `test_size=0.3` : assigns 30% of data for testing  
- `random_state=42` : ensures reproducible splits by fixing the random seed

---

## 7. ⚖️ Performing Feature Scaling

We will perform Min-Max scaling on our input features to ensure all the features are on the same scale. It ensures consistent scale and improves model performance. Scaling will be done on both training and testing data using the scaling parameters.

- `MinMaxScaler()` scales features to [0,1] range.  
- `fit_transform()` fits scaler on training data and transforms it.  
- `transform()` applies same scaler to test data.

---

## 8. 🧠 Building the Neural Network

We will build neural network using TensorFlow and Keras for regression. After building the model we have compiled the model using the Huber Loss because of the robustness of the regression against outliers.

- `keras.Sequential()` creates a stack of layers.  
- `Dense` layers are fully connected layers.  
- `activation='relu'` adds non-linearity.  
- Output layer uses `activation='linear'` because it’s regression.  
- **Huber loss** combines MSE and MAE advantages to handle outliers better.  
- `adam` optimizer adjusts weights efficiently.

---

## 9. 🏋️ Training the Model

We train the model on the scaled training data for 10 epochs with a batch size of 64, validating on the test set.

- `model.fit()` trains the model.  
- `epochs=10` means the model sees the whole data 10 times.  
- `batch_size=64` updates weights after every 64 samples.  
- `validation_data` evaluates model on test set during training.

---

## 10. 📈 Evaluating the Model

We predict scores on test data and evaluate model performance using mean absolute error (MAE).

---

## 11. 🖱️ Creating an Interactive Widget for Score Prediction

We build an interactive interface using ipywidgets so users can select match details and get a live predicted score.

- `widgets.Dropdown()` creates dropdown menus.  
- `widgets.Button()` creates a clickable button.  
- `predict_score()` function handles user inputs, encodes and scales them, runs prediction and displays result.  
- `display()` shows widgets in the notebook.
- 


---

## 🚀 How to Run This Project

Follow these steps to set up and run the IPL Score Prediction project on your local machine:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rohitlakha/ipl-score-prediction.git
cd ipl-score-prediction

### 2️⃣ Install Required Libraries

Make sure you have **Python 3.7+** installed. Then, install the required libraries:

```bash
pip install -r requirements.txt

### 3️⃣ Run the Notebook

```bash
jupyter notebook ipl_score_prediction.ipynb

##4️⃣ Use the Interactive Widget
Scroll to the bottom of the notebook and use the interactive UI to input match conditions and view live predictions.


## 📬 Author

**Rohit Lakha**  
🔗 [LinkedIn](https://www.linkedin.com/in/rohit-lakha/) • 🔗 [GitHub](https://github.com/Rohitlakha)

---

> ⭐ If you like this project, give it a star on GitHub and feel free to contribute!
