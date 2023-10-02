# classifier.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class ResumeClassifier:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)

    def train(self, dataset_path):
        data = pd.read_csv(r"data/UpdatedResumeDataSet.csv")
        X_train, X_test, y_train, y_test = train_test_split(data['Resume'], data['Category'], test_size=0.2, random_state=42)
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        self.clf.fit(X_train_tfidf, y_train)

    def predict(self, resume_text):
        resume_tfidf = self.tfidf_vectorizer.transform([resume_text])
        return self.clf.predict(resume_tfidf)[0]
