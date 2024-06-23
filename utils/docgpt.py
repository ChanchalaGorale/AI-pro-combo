import streamlit as st

import pdfplumber
from transformers import pipeline
import pytesseract
import nltk
import spacy
from langdetect import detect
import re
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
import openai
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.3.4_1/bin/tesseract'

nlp = spacy.load("en_core_web_sm")
ner = pipeline("ner")
nltk.download('punkt')
relationship_extractor = pipeline("text-classification", model="dslim/bert-base-NER")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


genai.configure(api_key="AIzaSyC0HQTg-oaShAG_0_GgxIqUCTTjtqRRK9E")

def pdf_to_text_and_images(pdf_path):
    text_content = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:
            text_content += page.extract_text()
           
            for img in page.images:
                img_obj = page.to_image().original.crop((img['x0'], img['top'], img['x1'], img['bottom']))
                text_content += pytesseract.image_to_string(img_obj)
                

    return text_content

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


    
def segment_sentences(text):
    return sent_tokenize(text)

# Tokenization
def tokenize_text(text):

    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()

    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens


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

def find_entities(text):
    doc = nlp(text)

    entities={}
 
    for ent in doc.ents:
        entities[ent.label_] = ent.text

    return entities


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def getgpt(set_page):
    st.button("← Go Back", on_click=set_page, args=['projects'])
    st.title("Document GPT")
    st.markdown("Upload your document")

    pdf = st.file_uploader("Only PDF files accepted.", type='pdf')

    if pdf is not None:
        #extract text
        text = pdf_to_text_and_images(pdf)

        summary=get_text_summary(text)

        if summary:
            st.title("Document Summary:")
            st.write(summary)

        sentences = segment_sentences(text)
        tokens = tokenize_text(". ".join(sentences)) 

        if tokens:
            st.write("Total tokens extracted: ",len(tokens))
        
        lang = detect(text)

        if lang != 'en':

            if st.button("Translate To English"):

                translation = translate_to_english(text)

                if translation:
                    st.write(translation)

        entities=find_entities(text)

        if entities:
            st.write("NER: ",entities)
        
        query = st.text_input("Ask a Question from the document")

        if query:
            OPENAI_API_KEY="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"
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
            )
            chunks = text_splitter.split_text(text=text)
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            VectorStore =FAISS.from_texts(chunks, embedding=embeddings)

            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            if response:
                st.write(response)
            else:
                st.write("Answer not Found! We suggest you upload relevant file to get the answer to this question.")

        # if user_question:
        #     user_input(user_question)
        
        #     text_chunks = get_text_chunks(text)
        #     get_vector_store(text_chunks)
        #     st.success("Done")

        
                    

