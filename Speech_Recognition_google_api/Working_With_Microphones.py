#Working With Microphones

### PyAudio will be used so, we need to install it 1st

! pip install pyaudio
import speech_recognition as sr
r=sr.Recognizer()
mic = sr.Microphone()
sr.Microphone.list_microphone_names()

#listen() to Capture Microphone Input

with mic as source:
    audio=r.listen(source)
r.recognizer_google(audio)

# To decrease noise in speech signal

with mic as source:
     r.adjust_for_ambient_noise(source)
     audio = r.listen(source)

# For some cases, Handling Unrecognizable Speech is important

