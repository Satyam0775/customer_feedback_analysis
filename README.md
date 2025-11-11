# 🧠 Customer Feedback Sentiment Analysis

**Author:** Satyam  
**Environment:** Python 3.10.18  
**Frameworks:** Scikit-learn, NLTK, Gradio  

---

## 🏗️ Project Overview
This project is an **end-to-end NLP system** that classifies customer reviews into **Positive**, **Negative**, or **Neutral** categories.  
It helps businesses analyze large volumes of customer feedback and understand sentiment patterns efficiently.  

The pipeline covers the complete ML workflow — from **data preprocessing** to **model training**, **evaluation**, and **deployment** using **Gradio** for an interactive interface.

---

## ⚙️ Technical Summary
- **Dataset:** Amazon Product Reviews (Kaggle)  
- **Language Processing:** NLTK (stopword removal, regex cleaning)  
- **Feature Engineering:** TF-IDF Vectorization (`max_features=5000`, `ngram_range=(1,2)`)  
- **Model:** Logistic Regression (`max_iter=200`)  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  
- **Deployment Tool:** Gradio  

---

## 🔍 Workflow Summary
1. **Data Cleaning**
   - Removed null values and duplicates  
   - Mapped numeric `Score` values into sentiment labels:  
     - `1–2 → Negative`  
     - `3 → Neutral`  
     - `4–5 → Positive`

2. **Feature Extraction**
   - Used TF-IDF to convert text data into numerical feature vectors  

3. **Model Training**
   - Trained a Logistic Regression classifier on TF-IDF vectors  
   - Saved model as `feedback_model.pkl` using Joblib  

4. **Evaluation**
   - Calculated Precision, Recall, and F1-score  
   - Visualized confusion matrix  

5. **Deployment**
   - Built and launched a Gradio web app for real-time predictions  
   - Input: customer review text  
   - Output: predicted sentiment label  

---

## 🌐 Run the App
Once dependencies are installed (`pip install -r requirements.txt`), launch the app:
```bash
python app.py
Then open the local URL (e.g. http://127.0.0.1:7860/
) in your browser.

📈 Future Enhancements

Add multilingual sentiment analysis (using multilingual BERT or translation APIs)

Implement auto-retraining on new feedback data

Integrate visualization dashboard (Streamlit / Power BI)

Explore advanced deep learning models (BERT, LSTM)

🧩 Key Tools Used
Component	Library / Framework
Data Handling	Pandas, NumPy
Text Preprocessing	NLTK
Feature Extraction	TF-IDF (Scikit-learn)
Model	Logistic Regression
Visualization	Matplotlib, Seaborn
Deployment	Gradio
Model Storage	Joblib
🏁 Conclusion

This project demonstrates a lightweight and effective NLP pipeline for sentiment classification.
It is easily extensible for real-world use and provides a strong baseline for further experiments with deep learning and multilingual data.
