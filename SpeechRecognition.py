import speech_recognition as sr
from pydub import AudioSegment

# Convertir un archivo de audio a formato compatible (si es necesario)
def convert_audio(file_path, output_format="wav"):
    audio = AudioSegment.from_file(file_path)
    output_path = f"{file_path.split('.')[0]}.{output_format}"
    audio.export(output_path, format=output_format)
    return output_path

# Reconocimiento de voz desde un archivo de audio
def recognize_from_file(file_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(file_path)
    with audio_file as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="es-ES")
        print("Texto reconocido:", text)
    except sr.UnknownValueError:
        print("No se pudo entender el audio")
    except sr.RequestError:
        print("No se pudo solicitar resultados al servicio de Google")

# Probar reconocimiento desde un archivo de audio
file_path = "ejemplo.mp3"  # Cambia esto por la ruta a tu archivo de audio
converted_file_path = convert_audio(file_path)
recognize_from_file(converted_file_path)