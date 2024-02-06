from gtts import gTTS
from TTS.api import TTS

def gtts(result):
    tts=gTTS(result)
    tts.save('files/result.mp3')  

def tts(result):
    model=TTS.list_models()[0]
    tts=TTS(model)
    tts.tts_to_file(text=result, speaker=tts.speakers[0],language=tts.languages[0],file_path='files/result.mp3')