# Customer Review Sentiment Analysis

This project implements a machine learning–based sentiment analysis system that classifies customer reviews into **Positive**, **Neutral**, or **Negative** categories using Natural Language Processing (NLP) techniques.

---

## 📌 Technologies Used
- Python
- NLP (Text Preprocessing)
- TF-IDF
- Logistic Regression
- Scikit-learn
- Pandas
- NumPy

---

## 🧠 Project Approach
1. Cleaned and preprocessed raw customer review text  
2. Removed stopwords and non-alphabetic characters  
3. Converted text into numerical features using TF-IDF  
4. Trained a Logistic Regression classifier  
5. Evaluated the model using accuracy and classification metrics  

---

## 📂 Project Structure
Customer-Review-Sentiment-Analysis/
├── data/
│ └── README.md
├── sentiment_analysis.py
└── README.md

---

## 📊 Dataset
Amazon Fine Food Reviews Dataset  
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews  

*Dataset is not uploaded due to size constraints.*

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn nltk
python sentiment_analysis.py
Observations:
- The model performs well on positive and negative reviews, with slightly lower performance on neutral reviews due to overlapping sentiment patterns.
- TF-IDF combined with Logistic Regression provides a good balance between accuracy and computational efficiency.
Results
The model achieves strong performance with good accuracy and balanced precision, recall, and F1-score across all sentiment classes.

