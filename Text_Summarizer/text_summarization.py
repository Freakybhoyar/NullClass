import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def count_words(text):
    return len(word_tokenize(text))

def get_summary_word_range(selected_range):
    ranges = {
        "50-100 words": (50, 100),
        "100-200 words": (100, 200),
        "200-400 words": (200, 400),
        "400-700 words": (400, 700),
        "700-1000 words": (700, 1000)
    }
    return ranges[selected_range]

def divide_text_into_sections(sentences):
    """Divide text into beginning, middle, and end sections."""
    total_sentences = len(sentences)
    section_size = total_sentences // 3
    
    beginning = sentences[:section_size]
    middle = sentences[section_size:2*section_size]
    end = sentences[2*section_size:]
    
    return beginning, middle, end

def score_sentence(sentence, section_type, position, total_positions):
    """Score sentences based on their importance and position."""
    score = 0
    words = word_tokenize(sentence.lower())
    
    # Basic content scoring
    important_words = set(words) - set(stopwords.words('english'))
    score += len(important_words) * 0.1
    
    # Position-based scoring
    if section_type == "beginning":
        if position == 0:  # First sentence of the text
            score += 5
        score += 3 * (1 - position/total_positions)  # Higher score for earlier sentences
    elif section_type == "middle":
        # Key plot development indicators
        plot_indicators = ["however", "but", "meanwhile", "later", "then", "after"]
        if any(indicator in sentence.lower() for indicator in plot_indicators):
            score += 2
        score += 2 * (1 - abs(0.5 - position/total_positions))  # Higher score for central sentences
    elif section_type == "end":
        if position == total_positions - 1:  # Last sentence
            score += 4
        score += 3 * (position/total_positions)  # Higher score for later sentences
    
    # Plot advancement indicators
    if any(word in sentence.lower() for word in ["discovers", "reveals", "learns", "finds"]):
        score += 2
    
    return score

def generate_balanced_summary(text, target_word_range):
    min_words, max_words = target_word_range
    input_word_count = count_words(text)
    
    if input_word_count < min_words:
        return None, f"Error: Input text has only {input_word_count} words. Cannot generate a {min_words}-{max_words} word summary."
    
    sentences = sent_tokenize(text)
    beginning, middle, end = divide_text_into_sections(sentences)
    
    # Score sentences in each section
    scored_sentences = []
    for section_type, section in [("beginning", beginning), ("middle", middle), ("end", end)]:
        for i, sent in enumerate(section):
            score = score_sentence(sent, section_type, i, len(section))
            scored_sentences.append((score, sent, section_type))
    
    # Sort sentences by score within each section
    scored_sentences.sort(reverse=True)
    
    # Allocate words proportionally to each section
    target_words = (min_words + max_words) // 2
    section_words = {
        "beginning": target_words * 0.33,
        "middle": target_words * 0.33,
        "end": target_words * 0.33
    }
    
    # Select sentences while maintaining section balance
    selected_sentences = []
    current_section_words = {"beginning": 0, "middle": 0, "end": 0}
    
    for score, sentence, section_type in scored_sentences:
        sentence_words = count_words(sentence)
        if current_section_words[section_type] + sentence_words <= section_words[section_type]:
            selected_sentences.append((sentences.index(sentence), sentence))
            current_section_words[section_type] += sentence_words
    
    # Sort selected sentences by their original position
    selected_sentences.sort(key=lambda x: x[0])
    summary = " ".join(sentence for _, sentence in selected_sentences)
    
    # Final word count check
    final_word_count = count_words(summary)
    if final_word_count < min_words:
        return None, f"Warning: Could only generate a {final_word_count}-word summary."
    
    return summary, None

# Streamlit UI
st.title("Text Summarization Tool")

text_input = st.text_area("Enter text to summarize:", height=200)
word_range_options = ["100-200 words","200-400 words","400-700 words","700-1000 words"]

selected_range = st.selectbox("Select summary length:", word_range_options)

if text_input.strip():
    input_word_count = count_words(text_input)
    st.info(f"Input text word count: {input_word_count}")

if st.button("Generate Summary"):
    if text_input.strip():
        target_range = get_summary_word_range(selected_range)
        summary, error = generate_balanced_summary(text_input, target_range)
        
        if error:
            st.error(error)
        else:
            st.subheader("Summary")
            st.write(summary)
            summary_word_count = count_words(summary)
            st.success(f"Summary word count: {summary_word_count}")
    else:
        st.warning("Please enter some text to summarize.")