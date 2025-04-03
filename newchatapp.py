import streamlit as st
import os
import time
import requests
from dotenv import load_dotenv
from pytube import YouTube
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq




# Load environment variables
load_dotenv()
ASSEMBLY_AI_KEY = os.getenv('75e4f4fd6233425f8712812fb6e4f3ed')
GROQ_API_KEY = os.getenv('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')

# AssemblyAI API details
ASSEMBLY_AI_URL = "https://api.assemblyai.com/v2"
HEADERS = {
    "authorization": ASSEMBLY_AI_KEY,
    "content-type": "application/json"
}

# Define the API base URL
base_url = "https://api.assemblyai.com/v2"

# Load API Key
api_token = os.getenv("ASSEMBLY_AI_KEY")

# Define Headers
headers = {
    "authorization": api_token,
    "content-type": "application/json"
}


import yt_dlp

def save_audio(url):
    """Downloads only the audio of a YouTube video using yt_dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': '/usr/bin/ffmpeg',  # Update this path if needed
        'quiet': True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "audio.mp3"
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None


def assemblyai_stt(audio_filename):
    """Uploads an audio file to AssemblyAI and retrieves the transcript."""
    with open(audio_filename, "rb") as f:
        response = requests.post(base_url + "/upload", headers=headers, files={"file": f})

    # Check if the upload was successful
    if response.status_code != 200:
        raise RuntimeError(f"Failed to upload file: {response.text}")

    upload_url = response.json().get("upload_url")
    
    # Submit for transcription
    transcript_request = {"audio_url": upload_url}
    transcript_response = requests.post(base_url + "/transcript", json=transcript_request, headers=headers)
    
    if transcript_response.status_code != 200:
        raise RuntimeError(f"Failed to start transcription: {transcript_response.text}")

    transcript_id = transcript_response.json()["id"]
    polling_url = f"{base_url}/transcript/{transcript_id}"

    while True:
        transcript_result = requests.get(polling_url, headers=headers).json()
        if transcript_result["status"] == "completed":
            break
        elif transcript_result["status"] == "failed":
            raise RuntimeError(f"Transcription failed: {transcript_result}")
        else:
            time.sleep(3)

    transcript_text = transcript_result["text"]
    
    # Save transcript to a file
    transcript_path = "transcription.txt"
    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(transcript_text)

    return transcript_path


from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
import os

from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings.openai import OpenAIEmbeddings  # ✅ Use the new API
import os

def chat_with_transcript(query):
    """Loads the transcript and allows querying using LangChain with OpenAI's new API."""
    transcript_path = "transcription.txt"

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file '{transcript_path}' not found. Ensure transcription is completed.")

    loader = TextLoader(transcript_path)

    # ✅ Use OpenAIEmbeddings correctly
    embeddings = HuggingFaceEmbeddings()  # ✅ Works without OpenAI API Key

    index = VectorstoreIndexCreator(embedding=embeddings).from_loaders([loader])
    
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, groq_api_key="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")  # Replace with your Groq API key
    return index.query(query, llm=llm)


# Streamlit UI
st.set_page_config(layout="wide", page_title="ChatAudio", page_icon="🔊")
st.title("Chat with Your Audio using LLM")

video_url = st.text_input("Enter the YouTube video URL")


if video_url:
    col1, col2 = st.columns(2)
   
    with col1:
        st.info("Your uploaded video")
        st.video(video_url)
        audio_file = save_audio(video_url)
        transcription = assemblyai_stt(audio_file)
        st.info(transcription)
   
    with col2:
        st.info("Chat Below")
        user_query = st.text_area("Ask your query here...")
        if user_query and st.button("Ask"):
            st.info("Your Query: " + user_query)
            answer = chat_with_transcript(user_query)
            st.success(answer)
