import os
from pytubefix import YouTube
import ffmpeg
from faster_whisper import WhisperModel
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class YoutubeDownloadAndTranscribe:
    def __init__(self):
        self.youtube = YouTube
        
    def _download_youtube_video(self, url: str):
        try:
            yt = self.youtube(url)
            
            audio_streams = yt.streams.filter(only_audio=True).first()
            
            if not os.path.exists('audios'):
                os.makedirs('audios')
            
            print(f"Downloading audio: {yt.title}...")
            output_file = os.path.join('audios', f"{yt.title}.m4a")
            audio_file = audio_streams.download(output_path='audios', filename_prefix="audio_")
            (ffmpeg.input(audio_file).output(output_file, acodec='aac').overwrite_output().run())
            
            print(f"Downloaded: {yt.title} to 'audios' folder")
            print(f"File path: {output_file}")
            return output_file
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            
    def _transcribe_audio(self, audio_path):
        try:
            print("Transcribing audio...")
            Device = "cuda" if torch.cuda.is_available() else "cpu"
            print(Device)
            model = WhisperModel("base.en", device="cuda" if torch.cuda.is_available() else "cpu")
            print("Model loaded")
            segments, info = model.transcribe(audio=audio_path, beam_size=5, language="en", max_new_tokens=128, condition_on_previous_text=False)
            segments = list(segments)
            extracted_texts = [[segment.text, segment.start, segment.end] for segment in segments]
            return extracted_texts
        except Exception as e:
            print(f"Transcription error: {e}")
            return []
            
    def download_and_transcribe(self, url: str):
        audio_file = self._download_youtube_video(url)
        transcriptions = self._transcribe_audio(audio_file)
        
        trans_text = ""
        for text, start, end in transcriptions:
            trans_text += (f"{text}")
        return trans_text
        
def main():
    yt = YoutubeDownloadAndTranscribe()
    transcript = yt.download_and_transcribe("https://www.youtube.com/watch?v=lv0LlujoNVQ")
    print(transcript)
    
if __name__=="__main__":
    main()