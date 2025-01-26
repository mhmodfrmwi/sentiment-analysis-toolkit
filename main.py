import tkinter as tk
from tkinter import messagebox, filedialog
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud
import threading

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

current_theme = "light"

def analyze_sentiment():
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

        root.after(0, lambda: display_results(results, positive_count, negative_count, input_text))

    threading.Thread(target=analyze, daemon=True).start()

def display_results(results, positive_count, negative_count, input_text):
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, "\n".join(results))

    for widget in chart_frame.winfo_children():
        widget.destroy()

    for widget in wordcloud_frame.winfo_children():
        widget.destroy()

    show_sentiment_chart(positive_count, negative_count)
    generate_wordcloud(input_text)
    update_overall_impressions(positive_count, negative_count)

def update_overall_impressions(positive_count, negative_count):
    overall_text = ""
    if positive_count > negative_count:
        overall_text = f"Overall Sentiment: 😊 Positive ({positive_count} Positive, {negative_count} Negative)"
    elif negative_count > positive_count:
        overall_text = f"Overall Sentiment: 😞 Negative ({positive_count} Positive, {negative_count} Negative)"
    else:
        overall_text = f"Overall Sentiment: 😐 Balanced ({positive_count} Positive, {negative_count} Negative)"

    overall_label.config(text=overall_text)

def clear_fields():
    text_entry.delete("1.0", tk.END)
    result_text.delete("1.0", tk.END)

    for widget in chart_frame.winfo_children():
        widget.destroy()

    for widget in wordcloud_frame.winfo_children():
        widget.destroy()

    overall_label.config(text="Overall Sentiment:")

def show_sentiment_chart(positive_count, negative_count):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Positive", "Negative"], [positive_count, negative_count], color=["green", "red"])
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud: Highlights of the Most Frequent Words")

    canvas = FigureCanvasTkAgg(fig, master=wordcloud_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

def save_results():
    results = result_text.get("1.0", tk.END).strip()
    if results:
        file = filedialog.asksaveasfile(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file:
            try:
                file.write(results)
                file.close()
                messagebox.showinfo("Saved", "Results saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")
    else:
        messagebox.showwarning("No Results", "No results to save.")

def toggle_dark_mode():
    global current_theme
    is_dark = current_theme == "dark"
    root.config(bg="black" if not is_dark else "white")
    theme_color = "white" if not is_dark else "black"

    for widget in root.winfo_children():
        if isinstance(widget, (tk.Text, tk.Button, tk.Label, tk.Frame)):
            widget.config(bg=theme_color, fg="black" if not is_dark else "white")

    current_theme = "dark" if not is_dark else "light"

root = tk.Tk()
root.title("Sentiment Analysis Tool")
root.geometry("800x900")
root.config(padx=20, pady=20)

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(main_frame)
scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

tk.Label(scrollable_frame, text="Sentiment Analysis Tool", font=('Helvetica', 18, 'bold')).pack(pady=10)

tk.Label(scrollable_frame, text="Enter text (one sentence per line):", font=('Helvetica', 12)).pack(pady=5)
text_entry = tk.Text(scrollable_frame, height=6, width=60, font=('Helvetica', 12))
text_entry.pack(pady=10)

button_frame = tk.Frame(scrollable_frame)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Analyze", command=analyze_sentiment, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Clear", command=clear_fields, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Save Results", command=save_results, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
tk.Button(button_frame, text="Toggle Dark Mode", command=toggle_dark_mode, bg="#9E9E9E", fg="white").pack(side=tk.LEFT, padx=5)

result_text = tk.Text(scrollable_frame, height=10, width=70, font=('Helvetica', 12), wrap=tk.WORD)
result_text.pack(pady=10)

overall_label = tk.Label(scrollable_frame, text="Overall Sentiment:", font=('Helvetica', 12, 'bold'))
overall_label.pack(pady=10)

chart_frame = tk.LabelFrame(scrollable_frame, text="Sentiment Chart", font=('Helvetica', 12), padx=10, pady=10)
chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)

wordcloud_frame = tk.LabelFrame(scrollable_frame, text="Word Cloud", font=('Helvetica', 12), padx=10, pady=10)
wordcloud_frame.pack(fill=tk.BOTH, expand=True, pady=5)

root.mainloop()
