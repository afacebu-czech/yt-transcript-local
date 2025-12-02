import os
from pytubefix import YouTube
import ffmpeg
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import logging
from translatepy.translators.google import GoogleTranslateV2
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

class TranscribeAndTranslate:
    def __init__(self):
        self.youtube = YouTube
        self.gtranslate = GoogleTranslateV2()
        
        
    # def test(self, audio_path):
    #     device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
    #     model_id = "openai/whisper-base.en"
    #     model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #         model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    #     )
    #     model.to(device)
        
    #     processor = AutoProcessor.from_pretrained(model_id)
        
    #     pipe = pipeline(
    #         "automatic-speech-recognition",
    #         model=model,
    #         tokenizer=processor.tokenizer,
    #         feature_extractor=processor.feature_extractor,
    #         dtype=torch_dtype,
    #         device=device,
    #         language="en",
    #     )
        
    #     result = pipe(audio_path, return_timestamps=True)
    #     print(result["text"])
    def _translate(self, text: str, translation_lang: str=None):
        if translation_lang:
            try:
                translated_text = str(self.gtranslate.translate(text, destination_language=translation_lang))
            except:
                translated_text = text
            return translated_text
        return text
        
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
        
    def _convert_float_to_seconds(self, float_var: float):
        # Less than 60 → treat as seconds
        if float_var < 60:
            sec = format(float_var, ".2f")
            return sec.replace(".", ":")

        # 60 or more → convert to minutes:seconds
        minutes = int(float_var // 60)
        seconds = float_var % 60  # remainder in seconds
        return f"{minutes}:{seconds:05.2f}".replace(".", ":")
    
    def _return_yt_html_embed(self, yt_url: str):
        video_id = yt_url
        if "?v=" in yt_url:
            video_id = yt_url.split("?v=")[-1].split("&")[0]
        elif "youtu.be/" in yt_url:
            video_id = yt_url.split("youtu.be/")[-1].split("?")[0]
        elif "/watch/" in yt_url:
            video_id = yt_url.split("/watch/")[-1].split("?")[0]
        
        HTML_str = (
            f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
            " </center>"
        )
        return HTML_str
        
    def _transcribe_only(self, audio_path: str):
        transcriptions = self._transcribe_audio(audio_path)
        
        transcribed_text = ""
        for text, start, end in transcriptions:
            transcribed_text += (f"{self._convert_float_to_seconds(start)} - {self._convert_float_to_seconds(end)}: {text}\n")
            
        return transcribed_text
    
    def _transcribe_and_translate(self, audio_path: str, language: str=None):
        transcriptions = self._transcribe_audio(audio_path)
        
        transcribed_and_translated_text = ""
        for text, start, end in transcriptions:
            transcribed_and_translated_text += (f"{self._convert_float_to_seconds(start)} - {self._convert_float_to_seconds(end)}: {self._translate(text, language)}\n")
            
        return transcribed_and_translated_text
    
    def from_youtube(self, task: str, path: str, language:str=None):
        if language == "None":
            language = None
        
        html_embed_url = self._return_yt_html_embed(path)
        audio_file_path = self._download_youtube_video(path)
        
        if task=="translate" and language:
            transcript = self._transcribe_and_translate(audio_file_path, language)
        else:
            transcript = self._transcribe_only(audio_file_path)
        
        return html_embed_url, transcript
        
    def from_audio_file(self, task: str, path: str, language: str=None):
        if language == "None":
            language = None
        
        if task=="translate" and language:
            transcript = self._transcribe_and_translate(path, language)
        else:
            transcript = self._transcribe_only(path)
            
        return transcript
        
def main():
    yt = TranscribeAndTranslate()
#    yt.test("https://www.youtube.com/watch?v=lv0LlujoNVQ")
    yt.test(r"C:\Users\PM SHIFT\Documents\czech-codes\yt-transcript-local\audios\audio_Why Some Projects Use Multiple Programming Languages.m4a")
    
if __name__=="__main__":
    main()
