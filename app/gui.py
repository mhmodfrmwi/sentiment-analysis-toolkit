"""Legacy Tkinter desktop GUI."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from wordcloud import WordCloud

from sentiment_toolkit.inference import get_analyzer
from sentiment_toolkit.preprocessing import split_sentences


class SentimentGUI:
    """Desktop interface for interactive sentiment analysis."""

    def __init__(self) -> None:
        self.analyzer = get_analyzer()
        self.theme = "light"
        self.root = tk.Tk()
        self.root.title("Sentiment Analysis Tool")
        self.root.geometry("900x950")
        self.root.config(padx=20, pady=20)
        self._build_layout()

    def _build_layout(self) -> None:
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda _: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(
            scrollable_frame,
            text="Sentiment Analysis Tool",
            font=("Helvetica", 18, "bold"),
        ).pack(pady=10)
        tk.Label(
            scrollable_frame,
            text=f"Model: {self.analyzer.model_id}",
            font=("Helvetica", 10),
        ).pack(pady=2)

        tk.Label(
            scrollable_frame,
            text="Enter text (one sentence per line):",
            font=("Helvetica", 12),
        ).pack(pady=5)
        self.text_entry = tk.Text(scrollable_frame, height=6, width=70, font=("Helvetica", 12))
        self.text_entry.pack(pady=10)

        button_frame = tk.Frame(scrollable_frame)
        button_frame.pack(pady=10)
        tk.Button(
            button_frame,
            text="Analyze",
            command=self._start_analysis,
            bg="#4CAF50",
            fg="white",
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame,
            text="Clear",
            command=self._clear_fields,
            bg="#f44336",
            fg="white",
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame,
            text="Save Results",
            command=self._save_results,
            bg="#2196F3",
            fg="white",
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            button_frame,
            text="Toggle Dark Mode",
            command=self._toggle_dark_mode,
            bg="#9E9E9E",
            fg="white",
        ).pack(side=tk.LEFT, padx=5)

        self.result_text = tk.Text(
            scrollable_frame,
            height=12,
            width=80,
            font=("Helvetica", 12),
            wrap=tk.WORD,
        )
        self.result_text.pack(pady=10)

        self.overall_label = tk.Label(
            scrollable_frame,
            text="Overall Sentiment:",
            font=("Helvetica", 12, "bold"),
        )
        self.overall_label.pack(pady=10)

        self.chart_frame = tk.LabelFrame(
            scrollable_frame,
            text="Sentiment Chart",
            font=("Helvetica", 12),
            padx=10,
            pady=10,
        )
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.wordcloud_frame = tk.LabelFrame(
            scrollable_frame,
            text="Word Cloud",
            font=("Helvetica", 12),
            padx=10,
            pady=10,
        )
        self.wordcloud_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    def _start_analysis(self) -> None:
        threading.Thread(target=self._analyze, daemon=True).start()

    def _analyze(self) -> None:
        input_text = self.text_entry.get("1.0", tk.END).strip()
        if not input_text:
            self.root.after(
                0,
                lambda: messagebox.showwarning("Input Error", "Please enter some text."),
            )
            return

        sentences = split_sentences(input_text)
        results = []
        label_counts: dict[str, int] = {}

        for sentence in sentences:
            prediction = self.analyzer.predict(sentence)
            label_counts[prediction.label] = label_counts.get(prediction.label, 0) + 1
            results.append(
                f"Sentence: {sentence}\n"
                f"Label: {prediction.label}\n"
                f"Confidence: {prediction.confidence:.1%}\n"
            )

        self.root.after(
            0,
            lambda: self._display_results(results, label_counts, input_text),
        )

    def _display_results(
        self,
        results: list[str],
        label_counts: dict[str, int],
        input_text: str,
    ) -> None:
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(results))

        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        for widget in self.wordcloud_frame.winfo_children():
            widget.destroy()

        self._show_chart(label_counts)
        self._show_wordcloud(input_text)
        self._update_overall(label_counts)

    def _show_chart(self, label_counts: dict[str, int]) -> None:
        labels = list(label_counts.keys())
        values = list(label_counts.values())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, values, color=["#2ecc71", "#e74c3c", "#95a5a6"][: len(labels)])
        ax.set_ylabel("Count")
        ax.set_title("Sentiment Distribution")
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

    def _show_wordcloud(self, text: str) -> None:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word Cloud")
        canvas = FigureCanvasTkAgg(fig, master=self.wordcloud_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

    def _update_overall(self, label_counts: dict[str, int]) -> None:
        if not label_counts:
            self.overall_label.config(text="Overall Sentiment:")
            return

        dominant = max(label_counts, key=label_counts.get)
        parts = ", ".join(f"{count} {label}" for label, count in label_counts.items())
        self.overall_label.config(text=f"Overall Sentiment: {dominant} ({parts})")

    def _clear_fields(self) -> None:
        self.text_entry.delete("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        for widget in self.wordcloud_frame.winfo_children():
            widget.destroy()
        self.overall_label.config(text="Overall Sentiment:")

    def _save_results(self) -> None:
        results = self.result_text.get("1.0", tk.END).strip()
        if not results:
            messagebox.showwarning("No Results", "No results to save.")
            return

        file = filedialog.asksaveasfile(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
        )
        if not file:
            return

        try:
            file.write(results)
            file.close()
            messagebox.showinfo("Saved", "Results saved successfully!")
        except OSError as exc:
            messagebox.showerror("Error", f"Failed to save results: {exc}")

    def _toggle_dark_mode(self) -> None:
        is_dark = self.theme == "dark"
        bg = "black" if not is_dark else "white"
        fg = "white" if not is_dark else "black"
        self.root.config(bg=bg)
        self.theme = "dark" if not is_dark else "light"

        for widget in self.root.winfo_children():
            if isinstance(widget, (tk.Text, tk.Button, tk.Label, tk.Frame, tk.LabelFrame)):
                try:
                    widget.config(bg=bg, fg=fg)
                except tk.TclError:
                    pass

    def run(self) -> None:
        self.root.mainloop()


def setup_gui() -> None:
    SentimentGUI().run()
