from gtts import gTTS
import pygame
import io

def text_to_speech(text="sorry I unable to hear you"):

    _tts_ = gTTS(text=text, lang='en')

    _speech_fp_ = io.BytesIO()
    _tts_.write_to_fp(_speech_fp_)
    _speech_fp_.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(_speech_fp_, 'mp3')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
