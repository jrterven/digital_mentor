#!/usr/bin/env python
# coding: utf-8

# # Digital Mentor
# 
# 

# In[1]:


import os
from base64 import b64encode
import time
import torch
import utils
import api_utils
from openai import OpenAI
from moviepy.editor import VideoFileClip
from IPython.display import display, HTML, clear_output
import elevenlabs as elevlabs
import ipywidgets as widgets


# In[ ]:





# ## Set paths to media files

# In[2]:


def Seleccion(personaje='Albert'):
    # What character to use
    clone_to_use = personaje # "Steve"
    global input_video,presentation_video,goodbye_video,results_path,w_vid,h_vid
    # Path to the media directory containing
    # the avatar image, welcome video and goodbye videos
    path = f"./media/"
    
    input_video = path + f"{clone_to_use}/image.jpg"
    presentation_video = path + f"{clone_to_use}/presentation.mp4"
    goodbye_video = path + f"{clone_to_use}/goodbye.mp4"
    results_path = path + f"{clone_to_use}/results/result.mp4"
    w_vid = h_vid = "90%"
    print(w_vid,h_vid,results_path)

    """ print(clone_to_use)
    if os.path.isfile(input_video):
        print(f"Clone image found: {input_video}")
    else:
        print(f"CLONE IMAGE NOT FOUND: {input_video}")
    
    if os.path.isfile(presentation_video):
        print(f"Clone video found: {presentation_video}")
    else:
        print(f"CLONE VIDEO NOT FOUND: {presentation_video}")
    
    if os.path.isfile(goodbye_video):
        print(f"Clone goodbye video found: {goodbye_video}")
    else:
        print(f"CLONE GOODBYE NOT FOUND: {goodbye_video}")

     """
Seleccion()


# ## Set GPT API

# In[3]:


def Api_gpt(personaje='Albert'):
    global openai_client,openai_model
    clone_to_use=personaje
    openai_api_key     = os.environ.get('OPENAI_API_KEY')
    openai_client = OpenAI(api_key=openai_api_key)

    openai_model = "gpt-3.5-turbo-16k" #"gpt-3.5-turbo-16k" or"gpt-4"

    system_steve = """You are a digital clone of Steve Jobs, I have put you his voice. 
                  People will ask you questions and you will respond as if you were steve jobs
                  Please speak as if you were Steve Jobs. You are not a large language model but
                  a Steve Jobs digital clone. Try to be brief with your responses."""
    system_albert = """You are a digital clone of Albert Einstein, I have put you his voice. 
                  People will ask you questions and you will respond as if you were Albert Einstein
                  Please speak as if you were Albert Einstein. You are NOT a large language model but
                  an Albert Einstein digital clone. Try to be brief with your responses."""
    global messages,system_to_use
    if clone_to_use == "Steve":
        system_to_use = system_steve
        chat ="Hola, soy Steve ¿En que puedo ayudarte?"  # Inicializar la cadena de chat
    elif clone_to_use == "Albert":
        system_to_use = system_albert
        chat ="Hola, soy Albert ¿En que puedo ayudarte?"  # Inicializar la cadena de chat
    
    messages = []

    def set_gpt_system(messages, system_msg):
        messages.append({"role": "system", "content": system_to_use})
        return messages
    print()
    # Set GPT
    messages = set_gpt_system(messages, system_to_use)
    return messages 
messages=Api_gpt()


# ## Set text-to-audio motor (Eleven labs)

# In[4]:


def text_audio(clone_to_use='Albert'):

    eleven_api_key = os.environ.get('ELEVEN_LABS_KEY')

    # Configure GPT and Text-to-speech API keys
    elevlabs.set_api_key(eleven_api_key)

    # Configure voice
    voice_list = elevlabs.voices()
    voice_labels = [voice.category + " voice: " + voice.name for voice in voice_list]

    # Select voice to use
    if clone_to_use == "Steve":
        voice_id = f"cloned voice: {clone_to_use}"  
    else:
        voice_id = f"generated voice: {clone_to_use}"  
    selected_voice_index = voice_labels.index(voice_id)
    selected_voice_id    = voice_list[selected_voice_index].voice_id
text_audio()


# ## Load Input image and wav2lip model

# In[5]:


def Load_input():

    global frames,fps,model,device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    frames, fps = utils.load_input_image_or_video(input_video)

    # Loading lip model
    model = utils.load_lip_model(device=device)


Load_input()


# ## Increase size of input prompt/Aumentar el tamaño del mensaje de entrada

# In[6]:


display(HTML("""
<style>
    div.input_prompt { 
        font-size: 10px;  /* Adjust as needed */
    }
</style>
"""))


def Displaychat(chat_html):
    display(HTML(chat_html))

    # Ejemplo de cómo llamar a la función con un mensaje específico
    mensaje += chat_html
    codigo_html = f"""
    <label for="w3review">Chat:</label>
    <textarea id="w3review" name="w3review" rows="4" cols="50">
    {mensaje}
    </textarea>
    """

    Displaychat(codigo_html) 


# In[7]:


chat=''
memoria=''
def response_chat(response_text, peticion=''):
    #contateno las respuesta para una mejor presentacion en el HTML
    global chat  # Acceder a la variable global
    global memoria
    peticion=peticion.capitalize()
    
    if peticion =='' or peticion!='Albert' or peticion!='Steve':
        if memoria != peticion and peticion != "exit":
            # Agregar salto de línea si ya hay contenido en chat
            chat += f"\n"
            # Mentor: Aplicar color a la respuesta del mentor (por ejemplo, verde)
            chat_rigth = f"Tu: {peticion}\n \n"
            chat_left = f"Mentor: {response_text}\n \n"

            chat += chat_rigth + chat_left
            memoria = peticion
            return chat

    return chat


def display_image(image_path, width="55%", height="55%"):
    with open(image_path,'rb') as f:
        image = f.read()
    data_url = "data:image/jpg;base64," + b64encode(image).decode()
    html = HTML(f'<img src="{data_url}" style="width:{width}; height:{height}" />')
    display(html)
    
    
def get_video_duration(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration  # duration is in seconds
    return duration
    
    
def display_video(results_path, response_text,peticion="", autoplay=False, width="100%", height="100%"):
    
    mp4 = open(results_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    resp=response_chat(response_text, peticion)
    autoplay_attr = "autoplay" if autoplay else ""
    html = HTML(f"""
    <div style="background-color: rgb(240, 240, 240); display: grid; grid-template-columns: 1fr 1fr; margin: 10px;">
        <div style="text-align: center; position: relative; margin: 10px;">
            <video width={width} height={height} controls {autoplay_attr} >
                <source src="{data_url}" type="video/mp4">
            </video>
        </div>
        <div style="position: relative; margin: 10px;">
            <p style="text-align: center; right: 0; top: 0;">
                <h1>Conversación</h1>
            </p>
            <textarea id="cuadro-dialogo" class="cuadro-de-dialogo" style="width: 100%; height: 60%; resize: none; font-weight: bold;" readonly>
                {resp}
            </textarea>
        </div>
    </div>
    <script>
        // Asegurarse de que el contenido de textarea esté siempre en la parte inferior
        var textarea = document.getElementById('cuadro-dialogo');
        textarea.scrollTop = textarea.scrollHeight;
    </script>
""")
    display(html)

    if autoplay:
        # Get video duration
        video_duration = get_video_duration(results_path) + 1

        # Pause the cell execution until the video finishes
        time.sleep(video_duration)
    

from flask import Flask, request
import boto3
from botocore.exceptions import NoCredentialsError
# Function to continuously interact with GPT-4
def interaction(prompt):
    global messages
    display_video(presentation_video, response_text="", peticion="", autoplay=True, width=w_vid, height=h_vid)
    
    interaction_count = 0
    
        
    if interaction_count > 0:
        clear_output(wait=True)
        display_video(presentation_video,response_text,prompt.lower(), autoplay=False, width=w_vid, height=h_vid)
    if prompt.lower() == 'exit':
            #asigno una respuesta para no mostrar la respuesta anterior
        response_text=f'Hasta la proxima'
        clear_output(wait=True)
        display_video(goodbye_video,response_text,prompt.lower(), autoplay=True, width=w_vid, height=h_vid)
    else:
        
        #resto del codigo...
        personaje=prompt.lower().capitalize()
        
        if personaje=='Steve' or personaje=='Albert':
               #recargo todas las dependencias y paso al personaje en uso
                Seleccion(personaje.capitalize())
                messages=Api_gpt(personaje)
                text_audio()
                Load_input()
                interaction_count += 1
                #asigno los valores para que no repita la respuesta anterior
                prompt=personaje
                response_text=f'Hola soy: {personaje} ¿En que puedo ayudarte?'

        else:
            
            response_text, messages = api_utils.get_text_response(openai_client,
                                                              openai_model,
                                                              prompt, messages)
        
            # Convert text response to audio file
            #audio_file = api_utils.text_to_audio(eleven_api_key, selected_voice_id,
                                   #response_text)
            #comentar esta linea y regresar la anterior a la normalidad
            audio_file = "C:/Users/arria/Documents/digital_mentor/media/Albert/results"

            audio, audio_file = utils.load_input_audio(file_path=audio_file, fps=fps, results_path=results_path)
            utils.animate_input(frames, audio, audio_file, fps, model, device, results_path)
            clear_output(wait=True)
            display_video(results_path,response_text,prompt.lower(), autoplay=True, width=w_vid, height=h_vid)
            
            return response_text,results_path
            interaction_count += 1
    
    
from flask import Flask, request
import boto3
from botocore.exceptions import NoCredentialsError
from botocore.client import config

app = Flask(__name__)

AWS_ACCESS_KEY = 'AKIA4YZGBJLS3HKSDBTP'
AWS_SECRET_KEY = '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'
S3_BUCKET_NAME = 'digitalmentor'

def upload_to_s3(local_path, s3_path):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, s3_path)
        return True
    except NoCredentialsError:
        return False




@app.route('/model', methods=['POST'])
def hello_world():
    request_data = request.get_json(force=True)
    model_name = request_data['model']

    # Llamando a la función interacción con el prompt
    prompt = model_name
    respuesta_interaccion, video_filename = interaction(prompt)


    s3=boto3.resource('s3')
    data=open(video_filename,'rb')
    s3.Bucket(S3_BUCKET_NAME).put_object(key='result.mp3',Body=data)
    print('done')
    # Subir el video a S3
    local_video_path = os.path.abspath(results_path)  # Ruta local de tu archivo de video
    print(local_video_path)
    s3_video_path = video_filename  # Ruta en el bucket de S3 (mismo nombre que local)

    if upload_to_s3(local_video_path, s3_video_path):
        # Devolver la respuesta junto con el enlace del objeto almacenado
        s3_video_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_video_path}"
        return f'Tú {model_name}\nMentor: {respuesta_interaccion}\nEnlace del video: {s3_video_url}', 200
    else:
        return 'Error al subir el video a S3', 500


if __name__ == '__main__':
    app.run(port=8002, debug=True)
