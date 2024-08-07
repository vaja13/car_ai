import speech_recognition as sr

def speech_to_text():

    _recognizer_ = sr.Recognizer()

    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        _recognizer_.adjust_for_ambient_noise(source)
        print("Listening...")

        try:
            audio = _recognizer_.listen(source, timeout=10, phrase_time_limit=30)

            print("Recognizing...")
            text = _recognizer_.recognize_google(audio)
            print("You said: " + text)
            return text
        except sr.WaitTimeoutError:
            print("No speech detected within the timeout period.")
        except sr.UnknownValueError:
            print("Google Web Speech API could not understand the audio")
        except sr.RequestError as e:
            print("Could not request results from Google Web Speech API; {0}".format(e))

