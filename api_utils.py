import requests
import tempfile
from openai import OpenAI
from elevenlabs import generate, play, set_api_key, voices, Models

# Function to get GPT response
def get_text_response(client, model, prompt, messages):
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    response_text = response.choices[0].message.content
    messages.append({"role": "assistant", "content": response_text})
    
    return response_text, messages


def text_to_audio(api_key, voice_id, response_text):
    # Convert response to audio
    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id

    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": api_key
    }

    data = {
      "text": response_text,
      "model_id" : "eleven_multilingual_v1",
      "voice_settings": {
        "stability": 0.4,
        "similarity_boost": 1.0
      }
    }

    response = requests.post(url, json=data, headers=headers)
    
    # Save audio data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
        f.flush()
        temp_filename = f.name
    return temp_filename