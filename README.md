# 📧 Spam Email Classifier (Machine Learning + NLP)

This project implements a Machine Learning based text classification system to detect whether an SMS message is **Spam or Not Spam (Ham)** using Natural Language Processing techniques.

---

## 🚀 Project Overview

The model is trained on the **SMS Spam Collection Dataset** and uses text preprocessing and TF-IDF feature extraction to convert textual data into numerical form.
A **Multinomial Naive Bayes classifier** is used to perform spam detection with high accuracy.

---

## 🎯 Features

* Text preprocessing (lowercasing, punctuation removal, stopword removal)
* TF-IDF vectorization for feature extraction
* Machine Learning classification using Naive Bayes
* Model evaluation using accuracy score and confusion matrix
* Custom message testing through command line interface

---

## 🧠 Machine Learning Concepts Used

* Natural Language Processing (NLP)
* Feature Engineering (TF-IDF)
* Supervised Learning
* Probabilistic Classification
* Model Evaluation Metrics

---

## 🛠️ Tech Stack

* Python
* Pandas
* Scikit-learn
* NLTK

---

## 📊 Dataset

SMS Spam Collection Dataset
Contains labeled SMS messages categorized as **Spam** or **Ham**.

---

## ⚙️ How to Run the Project

1. Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
```

2. Navigate to project folder:

```
cd spam-email-classifier
```

3. Install dependencies:

```
pip install pandas scikit-learn nltk
```

4. Run the program:

```
python spam_classifier.py
```

5. Enter a message to test spam prediction.

---

## ✅ Model Performance

* Achieved approximately **95–97% accuracy** on test dataset.
* Evaluated using confusion matrix and accuracy score.

---

## 📌 Future Improvements

* Compare multiple models (Logistic Regression, SVM)
* Deploy using Streamlit Web App
* Improve preprocessing using stemming/lemmatization
* Add visualization of model performance

---

## 👩‍💻 Author

Bhagyavi Vangapandu
Computer Science Engineering Student
