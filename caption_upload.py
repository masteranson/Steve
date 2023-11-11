import requests
import time
import subprocess
import json
import openai
import os
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
 
counter_file = 'counter.txt'

# Converts AAC file to WAV
def convert_aac_to_wav(aac_file_path, wav_file_path):

    # Load the AAC file
    audio = AudioSegment.from_file(aac_file_path, format="aac")
    
    # Export as WAV
    audio.export(wav_file_path, format="wav")

def read_counter(file_path):
    try:
        with open(file_path, 'r') as file:
            return int(file.read().strip())
    except FileNotFoundError:
        return 0

def write_counter(file_path, count):
    with open(file_path, 'w') as file:
        file.write(str(count))

client = OpenAI(api_key = OPENAI_API_KEY)
os.remove('output.aac')
# Define the FFmpeg command
command = [
    'ffmpeg', 
    '-i', 'rtmp://localhost/live/ZOOM', 
    '-t', '10', 
    '-vn', 
    '-acodec', 'copy', 
    'output.aac'
]

# Start the FFmpeg process
process = subprocess.Popen(command)

# Wait for the specified duration (2 seconds in this case)
time.sleep(11)

# Stop the process
process.terminate()

# Optionally, wait for the process to end
process.wait()

convert_aac_to_wav("output.aac", "output.wav")

#Add whisper processing here
audio_file = open("output.wav", "rb")
transcript = client.audio.translations.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)

system = "You are a converter that takes broken english and converts it to a normal sentence. Do not ouput anything other than the full sentence. "

system_prompt = {'role': 'system', 'content': system}
user_prompt = {'role': 'user', 'content': transcript}

response = client.chat.completions.create(
    model='gpt-4',
    messages=[system_prompt, user_prompt],
)
outputs = response.choices[0].message.content # type: ignore

print(outputs)

#split outputs into a list of sentences
sentences = outputs.split(".")

# Zoom Closed Captioning API URL and Parameters
zoom_cc_url = "https://wmcc.zoom.us/closedcaption"
meeting_id = "3292661088"  # Meeting ID
signature = "yXTYLaL6AgK9v9u84sLg7b4txOIKjMG_Mp49UEeRQTU.AG.U7VyvTmHYNLBl5BGAj0xP36wsixdZL_Tb3YU-cGCxlwE6Tp66Cg98Y1XF4_2Ffzs9csitlmLU5TtTjMcUA8LzkmnbZkVK0z7bzAHBonhGTgEXQ2b7hDQ-gqP_tvGpEkM-oQOkOv-qkxc9qQcrKBITj5RHqwgCps3F3mxvMl_aPrjOpfw0Bjj1PL7w8FleSngF0KQHUGQBjk.mW4zXwM6qP1oz2aVmWoX8w.zNPdltVUFClwLZUt"
ns = "WXVuZyBDaGFrIEFuc29uIFRzYW5nJ3MgUGVyc29u"  # ns value
expire = "108000"  # Expire value
lang = "en-US"  # Language code

# Function to send caption to Zoom
def send_caption(seq, caption):
    url = f"{zoom_cc_url}?id={meeting_id}&ns={ns}&expire={expire}&sparams=id%2Cns%2Cexpire&signature={signature}&seq={seq}&lang={lang}"
    headers = {
        'Content-Type': 'text/plain',
    }
    response = requests.post(url, headers=headers, data=caption.encode('utf-8'))
    return response

# List of captions to send
# captions = [
#     "Big COCK.",
#     "BIGGER COCK",
#     "BIGGEST COCK."
# ]
# captions = sentences
captions = [outputs,]

# Sending Captions

seq = read_counter(counter_file)
seq += 1

# print(f"Current counter value: {counter}")

for caption in captions:
    seq += 1
    response = send_caption(seq, caption)
    if response.status_code == 200:
        print(f"Caption {seq} sent successfully")
    else:
        print(f"Failed to send caption {seq}: {response.content}")
    time.sleep(1)  # Pause for a second between captions

write_counter(counter_file, seq)