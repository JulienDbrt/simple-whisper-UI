import streamlit as st
import os
from pydub import AudioSegment
import whisper

def convert_audio_to_wav(audio_file):
    """Convert an audio file to WAV format if it's not already in that format."""
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", audio_file.name)

    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())

    if not temp_path.lower().endswith('.wav'):
        audio = AudioSegment.from_file(temp_path)
        wav_file = temp_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_file, format='wav')
        return wav_file
    
    return temp_path

def transcribe_audio(audio_file, model_size="base"):
    # Convert audio to WAV if necessary
    audio_file = convert_audio_to_wav(audio_file)

    if not os.path.exists(audio_file):
        return f"File {audio_file} does not exist."

    # Load the specified model
    model = whisper.load_model(model_size)

    try:
        # Transcribe the audio file
        result = model.transcribe(audio_file)
        transcription = result["text"]
        return transcription
    except Exception as e:
        return str(e)

def main():
    st.title("Audio Transcription App")

    # File uploader
    uploaded_files = st.file_uploader("Upload WAV files", type=["wav", "mp3", "m4a", "ogg"], accept_multiple_files=True)

    # Model size selection
    model_size = st.selectbox("Select Model Size", ["tiny", "base", "small", "medium", "large"])

    # Button to process files
    if st.button("Process Files"):
        transcriptions = {}
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Display file name
                st.write("Processing:", uploaded_file.name)

                # Transcribe audio
                transcription = transcribe_audio(uploaded_file, model_size)

                # Save transcription in a dictionary with the file name
                transcriptions[uploaded_file.name] = transcription

                # Display transcription
                st.write("Transcription:")
                st.write(transcription)

            # After processing, clear the uploaded files
            uploaded_files.clear()

            # Button to download transcriptions
            if transcriptions:
                st.write("Download all transcriptions as a text file:")
                text_data = "\n\n".join([f"File: {name}\n{text}" for name, text in transcriptions.items()])
                st.download_button(label="Download Transcriptions", 
                                   data=text_data, 
                                   file_name="transcriptions.txt", 
                                   mime="text/plain")
        else:
            st.write("Please upload at least one audio file.")

if __name__ == "__main__":
    main()
