import numpy as np
import sqlite3
import pandas as pd
import zipfile
import io
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import speech_recognition as sr
import streamlit as st
from moviepy.editor import VideoFileClip
import tempfile
import os
import time

def extract_subtitles(db_path='eng_subtitles_database.db'):
    try:
        print("Extracting subtitles from database...")
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM zipfiles", conn)
        conn.close()

        def decode_binary(binary_data):
            with io.BytesIO(binary_data) as f:
                with zipfile.ZipFile(f, 'r') as zip_file:
                    return zip_file.read(zip_file.namelist()[0]).decode('latin-1')

        print("Decoding subtitle content and cleaning timestamps...")
        df['content'] = df['content'].apply(decode_binary)
        df['content'] = df['content'].apply(lambda x: re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', x))
        print(f"Extracted {len(df)} subtitle records.")
        return df
    except Exception as e:
        raise Exception(f"Error extracting subtitles: {str(e)}")

def preprocess_text(text):
    print("Preprocessing text...")
    text = text.lower().replace('\n', ' ').replace('\r', ' ')
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    print("Chunking text...")
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def vectorize_tfidf(texts):
    print("Vectorizing using TF-IDF...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    print("TF-IDF vectorization completed.")
    return tfidf_matrix, vectorizer

def vectorize_bert(texts):
    print("Vectorizing using BERT...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    print("BERT vectorization completed.")
    return embeddings

def store_embeddings(texts, db_name="chroma_subtitles"):
    print("Storing embeddings in ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=db_name)
    collection = chroma_client.get_or_create_collection(name="subtitles")
    for i, text in enumerate(texts):
        collection.add(ids=[str(i)], documents=[text])
    print(f"Stored {len(texts)} documents in ChromaDB.")
    return collection

def audio_to_text(audio_file):
    print("Converting audio to text...")
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            query_text = recognizer.recognize_google(audio_data)
            print(f"Transcribed audio query: {query_text}")
            return query_text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "There was an error with the speech recognition service."

def extract_audio_from_video(video_file):
    print("Extracting audio from video file...")
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    try:
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        print(f"Audio extracted to: {temp_audio_path}")
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None
    return temp_audio_path

def search_query(query, collection, vectorizer, tfidf_matrix, bert_embeddings=None, method="tfidf"):
    print(f"Executing search query: {query}")
    if method == "tfidf":
        query_vec = vectorizer.transform([query])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode([query], convert_to_numpy=True)
        similarity = cosine_similarity(query_embedding, bert_embeddings).flatten()

    top_indices = np.argsort(similarity)[::-1][:5]
    print(f"Top 5 similar subtitles found: {top_indices}")
    return top_indices, similarity[top_indices]

def get_top_subtitles(query, method='tfidf'):
    print(f"Getting top subtitles for query: {query}")
    df = extract_subtitles()

    df["content"] = df["content"].apply(preprocess_text)
    df["chunks"] = df["content"].apply(lambda x: chunk_text(x))

    all_chunks = [chunk for sublist in df["chunks"] for chunk in sublist]
    print(f"Total chunks for vectorization: {len(all_chunks)}")

    if method == 'tfidf':
        tfidf_matrix, vectorizer = vectorize_tfidf(all_chunks)
    else:
        bert_embeddings = vectorize_bert(all_chunks)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query])

    if method == 'tfidf':
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    else:
        similarity_scores = cosine_similarity(query_vector, bert_embeddings)

    df['similarity'] = similarity_scores.flatten()

    df_sorted = df.sort_values(by='similarity', ascending=False)

    print("Returning top 5 most relevant subtitles.")
    return df_sorted[['num', 'name', 'similarity']].head(5), df_sorted.iloc[0]

def main():
    st.title('Video Subtitle Search Engine')
    st.write("Welcome to the Video Subtitle Search Engine. You can search subtitles based on your query.")

    uploaded_file = st.file_uploader("Upload a Video or Audio File (2 minutes)", type=["wav", "mp3", "m4a", "mp4", "avi", "mov"])

    if uploaded_file:
        with st.spinner('Processing your file...'):
            time.sleep(2)
            if uploaded_file.type.startswith('audio'):
                query_text = audio_to_text(uploaded_file)
            elif uploaded_file.type.startswith('video'):
                audio_file_path = extract_audio_from_video(uploaded_file)
                if audio_file_path:
                    query_text = audio_to_text(audio_file_path)
                    os.remove(audio_file_path)
                else:
                    st.error("Failed to extract audio from video.")
                    return

            st.write(f"Transcribed Audio Query: {query_text}")

            method = st.selectbox('Choose Vectorization Method', ['bert', 'tfidf'])
            top_subtitles, most_relevant_subtitle = get_top_subtitles(query_text, method=method)

            st.write("Top 5 Most Relevant Subtitles:")
            st.dataframe(top_subtitles)

            st.write(f"Most Relevant Subtitle: {most_relevant_subtitle['name']}")
            st.text_area("Subtitle Content", most_relevant_subtitle['content'], height=300)

if __name__ == '__main__':
    main()
