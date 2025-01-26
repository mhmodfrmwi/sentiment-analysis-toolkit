import threading  
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import model_name

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text_entry, result_text, chart_frame, wordcloud_frame, overall_label, root):
    def analyze():
        input_text = text_entry.get("1.0", tk.END).strip()
        if not input_text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return

        sentences = input_text.split("\n")
        results = []
        positive_count, negative_count = 0, 0

        for sentence in sentences:
            if sentence.strip():
                inputs = tokenizer(sentence, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class = logits.argmax().item()
                class_labels = ["Negative", "Positive"]

                sentiment = class_labels[predicted_class]
                if sentiment == "Positive":
                    positive_count += 1
                else:
                    negative_count += 1
                sentiment_icon = "😊 Positive" if sentiment == "Positive" else "😞 Negative"
                results.append(f"Sentence: {sentence}\nAnalysis: {sentiment_icon}\n")

        root.after(0, lambda: display_results(results, positive_count, negative_count, input_text, result_text, chart_frame, wordcloud_frame, overall_label))

    threading.Thread(target=analyze, daemon=True).start()

def display_results(results, positive_count, negative_count, input_text, result_text, chart_frame, wordcloud_frame, overall_label):
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "\n".join(results))

    for widget in chart_frame.winfo_children():
        widget.destroy()

    for widget in wordcloud_frame.winfo_children():
        widget.destroy()

    show_sentiment_chart(positive_count, negative_count, chart_frame)
    generate_wordcloud(input_text, wordcloud_frame)
    update_overall_impressions(positive_count, negative_count, overall_label)

def update_overall_impressions(positive_count, negative_count, overall_label):
    overall_text = ""
    if positive_count > negative_count:
        overall_text = f"Overall Sentiment: 😊 Positive ({positive_count} Positive, {negative_count} Negative)"
    elif negative_count > positive_count:
        overall_text = f"Overall Sentiment: 😞 Negative ({positive_count} Positive, {negative_count} Negative)"
    else:
        overall_text = f"Overall Sentiment: 😐 Balanced ({positive_count} Positive, {negative_count} Negative)"

    overall_label.config(text=overall_text)

def show_sentiment_chart(positive_count, negative_count, chart_frame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Positive", "Negative"], [positive_count, negative_count], color=["green", "red"])
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

def generate_wordcloud(text, wordcloud_frame):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud: Highlights of the Most Frequent Words")

    canvas = FigureCanvasTkAgg(fig, master=wordcloud_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)