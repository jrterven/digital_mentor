
import os
import torch
import utils
import api_utils
from openai import OpenAI
import elevenlabs as elevlabs
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app, storage
import firebase_admin
import gradio as gr
from datetime import datetime, timedelta
from dotenv import load_dotenv



def seleccion(personaje='Albert', verbose=False):

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
    
    if verbose:
        print(w_vid,h_vid,results_path)

        print(clone_to_use)
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
    

seleccion(verbose=True)



def api_gpt(personaje='Albert'):



    # Load the environment variables from the .env file
    load_dotenv()
    global openai_client,openai_model
    clone_to_use=personaje
    openai_api_key = os.environ.get('OPENAI_KEY')
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
    #print(openai_client,openai_model,chat)

    def set_gpt_system(messages, system_msg):
        messages.append({"role": "system", "content": system_to_use})
        return messages
    # Set GPT
    messages = set_gpt_system(messages, system_to_use)

    return messages 
messages=api_gpt()

personaje_anterior = None


def text_audio(clone_to_use='Albert', verbose=False):
    global personaje_anterior, eleven_api_key

    eleven_api_key = os.environ.get('ELEVEN_KEY')
    # Check if this is the first time the code is executed or if the character is different from the previous one
    if personaje_anterior is None or clone_to_use != personaje_anterior:

        
        # Configure GPT and Text-to-speech API keys
        elevlabs.set_api_key(eleven_api_key)
        
        # Configure voice
        voice_list = elevlabs.voices()
        voice_labels = [voice.category + " voice: " + voice.name for voice in voice_list]

        if verbose:
            print("Existing voices:")
            print(voice_labels)

        # Select voice to use
        if clone_to_use == "Steve":
            voice_id = f"cloned voice: {clone_to_use}"  
        else:
            voice_id = f"generated voice: {clone_to_use}"  
        selected_voice_index = voice_labels.index(voice_id)
        selected_voice_id = voice_list[selected_voice_index].voice_id

        if verbose:
            print(f"\nSelected voice: {voice_id}")

        # Update the previous character
        personaje_anterior = clone_to_use
        
        return selected_voice_id
    else:
        return "No se ejecutó ningún proceso debido a que el personaje seleccionado es el mismo que el anterior."

# Using the feature to get the selected voice ID
selected_voice_id = text_audio(verbose=True)

def load_input():
    global frames,fps,model,device
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(f"Using {device}")
    frames, fps = utils.load_input_image_or_video(input_video)

    # Loading lip model
    model = utils.load_lip_model(device=device)

load_input()


def respuesta(prompt):
    global messages, selected_voice_id,response_text
    response_text, messages = api_utils.get_text_response(openai_client,
                                                                  openai_model,
                                                                  prompt, messages)
                # Convert text response to audio file
    audio_file = api_utils.text_to_audio(eleven_api_key, selected_voice_id,
                                       response_text)

    audio, audio_file = utils.load_input_audio(file_path=audio_file, fps=fps, results_path=results_path)
    utils.animate_input(frames, audio, audio_file, fps, model, device, results_path)

    

    return response_text

def interaction(prompt):
    
    global messages, selected_voice_id,response_text
    personaje=prompt.lower().capitalize()
    if personaje=='Steve' or personaje=='Albert':
        #recargo all dependencies and pass on to the character in use
        seleccion(personaje.capitalize())
        messages=api_gpt(personaje)
        selected_voice_id=text_audio(personaje)
        load_input()
        response_text=respuesta('Presentate en un renglon por favor')
        return response_text
    else:
        response_text=respuesta(prompt)
        return response_text
                

def subir_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate({
            
            "type": "service_account",
            "project_id": "mentores-c1064",
            "private_key_id": "d6d8e87281721ca7112d5e9df90631143decabce",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC1zw5ARQyZyDCi\nX25dujrP1SUNmcA/nZ77A+vXvHDX2JjsESs62qygADnSyuuaVW/sosFZwCN6puuO\nzFhQrsZCVNxb5X+IbPwyZVPnBc9s3GisuyD5NYOeQyTViJAji07TIoG/gwPx4Hzd\nwZAov9Uzs/s/TiNTc4i/dsWA1JSITE4FOKlrc0cx7zmF7fM9I+mxFJDbDLNuHr6t\nJPRlDTC5o5VAXuwa3AgIWluE+uHfJ8gVtXUs89QtvWdcJuOSYF5HjZ0+HvXVpGUO\n1zz/L/jcVw15l3LAIZb3fcHcAhHt9eq6xqrlV7Z4KXoBUF7tHy+tFIw3mXUCLeQ9\ns0Rb7HsHAgMBAAECggEAAzZjZBNUQ6tb4KKbCqDtxRpZC0J6OSWJ15mcIzW7MLUt\nDo7HGsUeY71dSDI5s4Jq9d1yWSmh9trYMq+9p79O1YE2W5FUjY6PbjyIHP7nSV8j\nolm0HXMqnjNFhVhiY55kiCqF7GJeQXnb+vcemvm4xK8bc2ToDuNtuSRaHQqkjTv0\nTO16tOjF2ktQpxK7kzpA556vzvPjlIy6xEJ+p9Uhy3DKI/IJ8938IJ5gr0aaUuxe\nxjRqY2MULP9KFbbvBIsdeRwToZpGGGYiwud4yNmul5ZeSOzlGClgFqWTu0Ul6SEc\nnKZphdNnPRWcVInXf93zXtZsZvLsN0wqMmYNvqooQQKBgQDbbLZXoQbRvSe7rjJR\nfGHBko9QkoC4Otp8EvvrsxZl113G+rrxjQn1MNGhF4UYUsKVYP3/moK/GWfc7G+g\nz5geeytOE6UJ+4pFDwUjKI9XPBWLapW2ADsmEYLAnLeP8Vs31KGVxUKKkW0Qx+wU\nMDLv3BnANc6e+LF3FLc0BkR5lwKBgQDUHTNByaeJjwyy0Fo3pdNe+Qdd/JC6G8O/\ngnKrVLgXP291rmNy1lh8YV7g+HM5Eqck31diKVRwlf/33ZV/AfqmU1Y1JoSZdwLg\niws0vOFOqjZ7lSgzAo9IqJjEC12zcfzpVcfAn6grWfJ7UavpFsyVauP5PoPV/QWa\nIkwPSp/YEQKBgH5e7eUp6DODLQ76FCC58dL6BW/x8BAqVQqAJHZqfcvJbUjbvi6/\ne6yqoRCV6yFHCKnfYmmDIynMB/VEdkW3SXTEmvwsdDB7nfaH3/2Prn4fLIlOqUpY\nd7hE/XjQySctacuNukH3iYsklhvECELYP33E1U/NrWIA+LQMSja8JVOhAoGAA+Zp\ni9seVnwn2p3UGtPUuBlSFltPeeyKw9mtLBNJszu6W+qR74mbZOYRbYeD7te19Qqa\nO7bQ06UeaLtNRWGO70H3AtErfPrgNaq40QZsChs9Fzad35o7cjWPYYNn/KWq5ctq\n+dK1r95eg//zbjy6FEE74dhRajzVvojE5z8TA1ECgYEAkHBpU0a5JSX1NTW0PpWC\nzyhRQ6Bi4fB/gn7tbyvyPgeNYf4hatuuI5X3WOAVSm3NcU16ZsxJinKPd0bpiAAi\nPsSGaaF1FZLXLTuUPNUHRIoBImHbSFyWOsgdCY11y2yqdM0cyIruvAErcDsir1NN\nZrMdNVK2QvZMJd9hc4kkzL0=\n-----END PRIVATE KEY-----\n",
            "client_email": "firebase-adminsdk-tz59u@mentores-c1064.iam.gserviceaccount.com",
            "client_id": "117993452038395169690",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-tz59u%40mentores-c1064.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        })
        initialize_app(cred, {'storageBucket': 'mentores-c1064.appspot.com'})

    # Initialize Firestore
    db = firestore.client()
    coleccion_ref = db.collection('Pruebas')

    # Logic to upload the file and get the tokenized URL
    archivo_ruta = results_path.lstrip('./')
    bucket = storage.bucket()
    blob = bucket.blob(archivo_ruta)
    blob.upload_from_filename(archivo_ruta)
    hora_expiracion = datetime.utcnow() + timedelta(minutes=5)
    token = blob.generate_signed_url(expiration=hora_expiracion, method='GET')
    archivo_url_con_token = token

    datos = {
        'archivo_url': archivo_url_con_token,
    }
    coleccion_ref.add(datos)
    return archivo_url_con_token

def mostrar_video_con_texto(Request):
    if not Request:
        return "Por favor, completa ambos campos."
    respuesta = interaction(Request)
    
    URL_VIDEO = subir_firebase()
    print(URL_VIDEO)
    reproductor_video = """<video width="640" height="480" controls autoplay>
                            <source src="{}" type="video/mp4">
                            Your browser does not support the video tag.
                          </video>""".format(URL_VIDEO)
    
    # Alineación del texto a la derecha del video con un poco de separación
    Request = '<div style="float:left; padding-right:20px;">{}</div>'.format(Request.replace("\n", "<br>"))
    respuesta = '<div style="float:right; padding-left:20px;">{}</div>'.format(respuesta.replace("\n", "<br>"))
    
    # Combinar el reproductor de video y el texto
    contenido = '<div style="overflow:auto;">{}<br>{}<br>{}</div>'.format(reproductor_video, Request, respuesta)
    return contenido

interfaz = gr.Interface(fn=mostrar_video_con_texto, inputs="text", outputs="html", title="Mentores Digitales", allow_flagging=False)
interfaz.launch()


