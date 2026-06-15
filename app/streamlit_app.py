"""Streamlit web interface for sentiment analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from sentiment_toolkit.inference import get_analyzer
from sentiment_toolkit.preprocessing import read_csv_texts, split_sentences

st.set_page_config(page_title="Sentiment Analysis Toolkit", page_icon="📊", layout="wide")

st.title("Sentiment Analysis Toolkit")
st.caption("Transformer-based sentiment classification with confidence scores.")

analyzer = get_analyzer()
st.sidebar.markdown(f"**Model:** `{analyzer.model_id}`")

tab_single, tab_batch, tab_file = st.tabs(["Single text", "Multi-line", "CSV upload"])

with tab_single:
    text = st.text_area("Enter text", height=120, placeholder="I love this product!")
    if st.button("Analyze", key="single"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            result = analyzer.predict(text)
            col1, col2 = st.columns(2)
            col1.metric("Sentiment", result.label)
            col2.metric("Confidence", f"{result.confidence:.1%}")
            st.json(result.scores)

with tab_batch:
    batch_text = st.text_area(
        "Enter one sentence per line",
        height=160,
        placeholder="Great service\nTerrible experience",
    )
    if st.button("Analyze lines", key="batch"):
        sentences = split_sentences(batch_text)
        if not sentences:
            st.warning("Please enter at least one sentence.")
        else:
            results = [analyzer.predict(sentence) for sentence in sentences]
            df = pd.DataFrame(
                [
                    {
                        "text": item.text,
                        "label": item.label,
                        "confidence": item.confidence,
                    }
                    for item in results
                ]
            )
            st.dataframe(df, use_container_width=True)

            counts = df["label"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            counts.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c", "#95a5a6"][: len(counts)])
            ax.set_title("Sentiment distribution")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            combined_text = " ".join(df["text"].tolist())
            if combined_text.strip():
                wordcloud = WordCloud(width=800, height=300, background_color="white").generate(
                    combined_text
                )
                fig_wc, ax_wc = plt.subplots(figsize=(8, 3))
                ax_wc.imshow(wordcloud, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

with tab_file:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    text_column = st.text_input("Text column name", value="text")
    if uploaded and st.button("Analyze CSV", key="file"):
        records = read_csv_texts(uploaded.getvalue(), text_column=text_column)
        rows = []
        for record in records:
            result = analyzer.predict(record.text)
            rows.append(
                {
                    "row_index": record.index,
                    "text": result.text,
                    "label": result.label,
                    "confidence": result.confidence,
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download results",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_results.csv",
            mime="text/csv",
        )
