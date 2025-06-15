import requests
import os
from dotenv import load_dotenv

load_dotenv()
SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
SPEECH_ENDPOINT = os.getenv("AZURE_SPEECH_ENDPOINT")  # e.g. https://your-resource.cognitiveservices.azure.com/

def azure_tts(text, voice="en-US-JennyNeural"):
    tts_url = SPEECH_ENDPOINT.rstrip("/") + "/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3"
    }
    ssml = f"""
    <speak version='1.0' xml:lang='en-US'>
        <voice xml:lang='en-US' name='{voice}'>{text}</voice>
    </speak>
    """
    resp = requests.post(tts_url, headers=headers, data=ssml.encode('utf-8'))
    if resp.status_code == 200:
        return resp.content
    return None

def azure_stt(audio_bytes, language="en-US"):
    stt_url = SPEECH_ENDPOINT.rstrip("/") + f"/speech/recognition/conversation/cognitiveservices/v1?language={language}"
    headers = {
        "Ocp-Apim-Subscription-Key": SPEECH_KEY,
        "Content-Type": "audio/wav"
    }
    resp = requests.post(stt_url, headers=headers, data=audio_bytes)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("DisplayText", None)
    return None
