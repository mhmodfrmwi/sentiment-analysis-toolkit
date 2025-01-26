# Sentiment Analysis Toolkit

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive solution for sentiment analysis combining transformer model training(using fine-tuning) and an interactive GUI application.

## Features ✨

- Fine-tuned GPT-2 model for sentiment analysis
- Interactive desktop GUI with visual analytics
- Real-time sentiment prediction
- Word cloud visualization
- Light/dark mode toggle
- Batch text processing
- Results export capability

## Installation

### Requirements

- Python 3.8+
- pip package manager

1. Clone the repository:

```bash
git clone https://github.com/mhmodfrmwi/sentiment-analysis-toolkit.git
cd sentiment-analysis-toolkit
```

2. Install required packages:

# For model training

pip install datasets transformers evaluate

# For GUI application

pip install matplotlib wordcloud

3. Model training :
   Upload the training notebook (model.ipynb) to Google Colab

   Enable GPU acceleration (Runtime → Change runtime type → GPU)

   Run all cells in sequence

4. After training, download the model:
   model.save_pretrained('./model')
   tokenizer.save_pretrained('./model')

### Usage

To start the GUI application, run the following command:
python main.py
