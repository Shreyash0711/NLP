import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load datasets
emotion_df = pd.read_csv("combined_emotion.csv")
sentiment_df = pd.read_csv("combined_sentiment_data.csv")

# Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Emotion model
X_emotion = vectorizer.fit_transform(emotion_df['sentence'])
y_emotion = emotion_df['emotion']
emotion_model = LogisticRegression(max_iter=1000)
emotion_model.fit(X_emotion, y_emotion)

# Sentiment model
X_sentiment = vectorizer.transform(sentiment_df['sentence'])
y_sentiment = sentiment_df['sentiment']
sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_sentiment, y_sentiment)

# GUI setup
class EmotionSentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion and Sentiment Predictor")
        self.root.geometry("700x500")

        self.label = tk.Label(root, text="Enter your text:", font=('Arial', 14))
        self.label.pack(pady=10)

        self.text_entry = tk.Text(root, height=5, width=70)
        self.text_entry.pack(pady=10)

        self.predict_btn = tk.Button(root, text="Predict", command=self.predict, font=('Arial', 12))
        self.predict_btn.pack(pady=10)

        self.result_frame = tk.Frame(root)
        self.result_frame.pack(fill='both', expand=True)

        self.result_label = tk.Label(self.result_frame, text="", font=('Arial', 14))
        self.result_label.pack(pady=10)

        self.canvas = None

    def predict(self):
        input_text = self.text_entry.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showerror("Input Error", "Please enter some text to analyze.")
            return

        input_vector = vectorizer.transform([input_text])
        emotion_pred = emotion_model.predict(input_vector)[0]
        sentiment_pred = sentiment_model.predict(input_vector)[0]

        self.result_label.config(text=f"Emotion: {emotion_pred} | Sentiment: {sentiment_pred}")

        # Show chart
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(['Emotion', 'Sentiment'], [1, 1], tick_label=[emotion_pred, sentiment_pred], color=['skyblue', 'lightcoral'])
        ax.set_title("Prediction Overview")
        self.canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionSentimentApp(root)
    root.mainloop()
