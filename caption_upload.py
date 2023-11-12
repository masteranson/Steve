import os
import re
import time
import signal
import requests
import subprocess
import signal
import openai
import os
from pynput import keyboard
from openai import OpenAI
from pydub import AudioSegment
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

counter_file = 'counter.txt'

ffmpeg_process = None

# Zoom Closed Captioning API URL and Parameters
zoom_link = "https://wmcc.zoom.us/closedcaption?id=94993538458&ns=QW1vZ2ggSm9zaGkncyBab29tIE1lZXRpbmc&expire=108000&sparams=id%2Cns%2Cexpire&signature=ZwTwDgxMhlrAP2Hl12fl3uBbf_QluSlCSfLVVEAVHmY.AG.hX8MelrhKsM80HVmGvvzxvMAGH7sPcj26s5KFvsOq9S5UFDUOoHCBQtN1v5sGP3lzEZiPSYi14Wbx62GNKCb0IpHnNU57qmbEvSVk85LSXCfobxyJXTf8jW7afcmiwEXCnrPhIQj9mp7OH56PQXAF1UbR-3SaMxoKAuJ149T3-TvZwaUZFzZvrxiLHrcJLGFWr3U.YpQskyk6f4xt9XgZoB0RLw.WeE13Zyit8QBt1Ze"
zoom_cc_url = "https://wmcc.zoom.us/closedcaption"
lang = "en-US"  # Language code
ns = "WXVuZyBDaGFrIEFuc29uIFRzYW5nJ3MgUGVyc29u"  

# parse the zoom link for the id, ns, signature, and expire

find_id = re.search(r'id=(\d+)', zoom_link)
meeting_id = find_id.group(1)
find_ns = re.search(r'ns=(\w+)', zoom_link)
ns = find_ns.group(1)
find_expire = re.search(r'expire=(\d+)', zoom_link)
expire = find_expire.group(1)
find_signature = re.search(r'signature=(\w+)', zoom_link)
signature = find_signature.group(1)


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

# Function to send caption to Zoom
def send_caption(seq, caption):
    url = f"{zoom_cc_url}?id={meeting_id}&ns={ns}&expire={expire}&sparams=id%2Cns%2Cexpire&signature={signature}&seq={seq}&lang={lang}"
    headers = {
        'Content-Type': 'text/plain',
    }
    response = requests.post(url, headers=headers, data=caption.encode('utf-8'))
    return response

def start_ffmpeg():
    command = [
        'ffmpeg', 
        '-i', 'rtmp://localhost/live/ZOOM', 
        '-vn', 
        '-acodec', 'copy', 
        'output.aac'
    ]
    return subprocess.Popen(command)

client = OpenAI(api_key = OPENAI_API_KEY)
if os.path.exists('output.aac'):
    os.remove('output.aac')

duration = 10 #Recording time

# Define the FFmpeg command
command = [
    'ffmpeg', 
    '-i', 'rtmp://localhost/live/ZOOM', 
    '-vn', 
    '-acodec', 'copy', 
    'output.aac'
]

# Start the FFmpeg process
process = subprocess.Popen(command)

def stop_ffmpeg(process):
    process.send_signal(signal.SIGINT)
    process.wait()
    print("Recording stopped and file saved.")

def on_press(key):
    global ffmpeg_process
    try:
        if key.char == 't':  # Start recording on 't' key press
            if ffmpeg_process is None:  # Start FFmpeg if it's not already running

                if os.path.exists("output.aac"):
                    os.remove("output.aac")

                ffmpeg_process = start_ffmpeg()
    except AttributeError:
        pass

def on_release(key):
    global ffmpeg_process
    if key.char == 't':  # Stop recording on 't' key release
        if ffmpeg_process is not None:
            time.sleep(0.1)  # Wait for 0.1 seconds
            stop_ffmpeg(ffmpeg_process)
            ffmpeg_process = None  # Reset the process

        #Add the whisper and openai processing here
        convert_aac_to_wav("output.aac", "output.wav")
        #Add whisper processing here
        client = OpenAI(api_key = OPENAI_API_KEY)
        audio_file = open("output.wav", "rb")
        transcript = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file, 
        response_format="text"
        )

        print('----------------------')
        print(transcript)
        print('----------------------')

        system_prompt = {'role': 'system', 'content': system}
        user_prompt = {'role': 'user', 'content': transcript}

        response = client.chat.completions.create(
            model='gpt-4',
            messages=[system_prompt, user_prompt],
        )
        outputs = response.choices[0].message.content
        print('----------------------')
        print(outputs)
        print('----------------------')
        seq = read_counter(counter_file)
        seq += 1

        response = send_caption(seq, outputs)
        if response.status_code == 200:
            print(f"Caption {seq} sent successfully")
        else:
            print(f"Failed to send caption {seq}: {response.content}")
        time.sleep(1)  # Pause for a second between captions

        write_counter(counter_file, seq)

        
        if key == keyboard.Key.esc:
            return False  # Stop the listener

# Collect events until released
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()




