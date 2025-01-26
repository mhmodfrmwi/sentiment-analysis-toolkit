import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from sentiment_analysis import analyze_sentiment, display_results, show_sentiment_chart, generate_wordcloud, update_overall_impressions
from config import current_theme

def clear_fields(text_entry, result_text, chart_frame, wordcloud_frame, overall_label):
    text_entry.delete("1.0", tk.END)
    result_text.delete("1.0", tk.END)

    for widget in chart_frame.winfo_children():
        widget.destroy()

    for widget in wordcloud_frame.winfo_children():
        widget.destroy()

    overall_label.config(text="Overall Sentiment:")

def save_results(result_text):
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

def toggle_dark_mode(root, current_theme):
    is_dark = current_theme == "dark"
    root.config(bg="black" if not is_dark else "white")
    theme_color = "white" if not is_dark else "black"

    for widget in root.winfo_children():
        if isinstance(widget, (tk.Text, tk.Button, tk.Label, tk.Frame)):
            widget.config(bg=theme_color, fg="black" if not is_dark else "white")

    current_theme = "dark" if not is_dark else "light"

def setup_gui():
    root = tk.Tk()
    root.title("Sentiment Analysis Tool")
    root.geometry("800x900")
    root.config(padx=20, pady=20)

    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(main_frame)
    scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

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

    tk.Button(button_frame, text="Analyze", command=lambda: analyze_sentiment(text_entry, result_text, chart_frame, wordcloud_frame, overall_label, root), bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Clear", command=lambda: clear_fields(text_entry, result_text, chart_frame, wordcloud_frame, overall_label), bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Save Results", command=lambda: save_results(result_text), bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Toggle Dark Mode", command=lambda: toggle_dark_mode(root, current_theme), bg="#9E9E9E", fg="white").pack(side=tk.LEFT, padx=5)

    result_text = tk.Text(scrollable_frame, height=10, width=70, font=('Helvetica', 12), wrap=tk.WORD)
    result_text.pack(pady=10)

    overall_label = tk.Label(scrollable_frame, text="Overall Sentiment:", font=('Helvetica', 12, 'bold'))
    overall_label.pack(pady=10)

    chart_frame = tk.LabelFrame(scrollable_frame, text="Sentiment Chart", font=('Helvetica', 12), padx=10, pady=10)
    chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    wordcloud_frame = tk.LabelFrame(scrollable_frame, text="Word Cloud", font=('Helvetica', 12), padx=10, pady=10)
    wordcloud_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    root.mainloop()