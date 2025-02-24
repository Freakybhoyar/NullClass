# Text Summarization Tool

A simple Streamlit-based text summarization tool that balances information across the beginning, middle, and end sections of the text.

## Features
- Uses **NLTK** for sentence tokenization and scoring.
- Generates summaries within a user-selected word range.
- Implements importance-based sentence scoring.
- Easy-to-use **Streamlit UI**.

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/Freakybhoyar/NullClass.git
cd NullClass
```

### 2. Create a virtual environment (Optional but recommended)
```bash
python -m venv summarizer_env
source summarizer_env/bin/activate  # Mac/Linux
summarizer_env\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
streamlit run app.py
```
