
import pdfplumber
import pytesseract

import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from transformers import pipeline
import nltk
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize

from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer


nltk.download('punkt')
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract'
nlp = spacy.load("en_core_web_sm")
ner = pipeline("ner")
relationship_extractor = pipeline("text-classification", model="dslim/bert-base-NER")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")



def pdf_to_text_and_images(pdf_path):
    text_content = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:
            text_content += page.extract_text()
           
            for img in page.images:
                img_obj = page.to_image().original.crop((img['x0'], img['top'], img['x1'], img['bottom']))
                text_content += pytesseract.image_to_string(img_obj)
                

    return text_content


def extract_key_value_pairs(text):
    # A simple regex-based approach for key-value pair extraction
    key_value_pattern = re.compile(r'(\b\w+):\s*(.+)')
    key_value_pairs = dict(re.findall(key_value_pattern, text))
    return key_value_pairs

def clean_text(text):
    # Remove special characters, punctuation, and formatting
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

    # Normalize to lowercase
    text = text.lower()

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text

# Sentence Segmentation
def segment_sentences(text):
    return sent_tokenize(text)

# Tokenization
def tokenize_text(text):
    tokens = []
    for i in word_tokenize(text):
        tokens.append(i)
    return tokens

def find_entities(text):
    doc = nlp(text)

    entities={}
 
    for ent in doc.ents:
        entities[ent.label_] = ent.text

    return entities

def translate_to_english(text,):
    lang = detect(text)
    if lang == 'en':
        return text
    model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]



def get_text_summary(text):    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    try:
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        
        return summary[0]['summary_text']

    except IndexError as e:
        max_length=1024
        words = text.split()
        chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
        summaries = []


        for chunk in chunks:
            try:
                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])

            except IndexError as e:
                print("An error occurred with a chunk:", str(e))
        
        
        summary = " ".join(summaries)


        return summary