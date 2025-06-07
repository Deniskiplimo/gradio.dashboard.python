import pyttsx3

try:
    engine = pyttsx3.init()
    print("Engine initialized successfully")
    engine.say("Hello, this is a test")
    engine.runAndWait()
    print("Speech done")
except Exception as e:
    print("Error:", e)
