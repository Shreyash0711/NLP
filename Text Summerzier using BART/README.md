# 📰 Text Summarization using BART

A NLP & deep learning project that uses **Facebook's BART-large** model to generate abstractive summaries from news articles. The model is applied to the **CNN/DailyMail dataset**, and integrated with a real-time **Gradio UI** for user interaction and evaluation.

---

## 📌 Project Highlights

- ✅ **Model**: BART-large (Transformer-based, pre-trained by Facebook AI)
- 📚 **Dataset**: CNN/DailyMail (287,113 train | 11,490 test)
- 💬 **Summarization Type**: Abstractive (generates new text, not just extracts)
- 📊 **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L (up to 43.9%)
- 🌐 **Interface**: Gradio for real-time article summarization

---

## 🧠 Technologies Used

| Tool/Library     | Purpose                         |
|------------------|----------------------------------|
| `transformers`   | BART model + tokenizer           |
| `pandas`         | Data loading + cleaning          |
| `nltk`           | Tokenization and text prep       |
| `rouge-score`    | Summarization evaluation         |
| `gradio`         | User interface for live testing  |

---

## 🧪 How It Works

1. **Load Dataset**: CNN/DailyMail dataset from KaggleHub
2. **Preprocess Text**: Clean long articles using regex & NLP
3. **Summarize**: Use BART-large-CNN to generate short summaries
4. **Evaluate**: Use ROUGE metrics to compare with human-written summaries
5. **Interface**: Launch a Gradio app for real-time summarization

---

## 🖥️ Gradio Interface

```bash
▶ Paste article text  
▶ Click "Generate Summary"  
▶ Get a clean, abstractive summary under 60 words
```

---

## 📊 Example Output
- Input (truncated):

A Russian man fed parrots, guinea pigs, and puppies to his pet boa snake...

- Human Summary:

Andrei Generalov uploads videos of his Boa Constrictor eating pets...

- BART Summary:

Andrei Generalov uploaded videos of his boa 'King' devouring pets...

- ROUGE-L: 43.9%


