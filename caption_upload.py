import requests
import time

# Zoom Closed Captioning API URL and Parameters
zoom_cc_url = "https://wmcc.zoom.us/closedcaption"
meeting_id = "3292661088"  # Meeting ID
signature = "HWUh09sSNYKyH1cJHz9jhznJQogD3ZrxCFIC851fuyo.AG.g5Ml4UGd5sLArgdEZuBRz1Y-HN-ksMcCCjiUbJYpnvZ44h2_CuVIHPXtHUehkobITTyJqNNH7A1-KMp0DgpZPHxv_U2JLWRtv4xpsZ3J7dFMwkVMo8U1QS3Qic9yhGBA13qrLDIstKki-Yg1h-cqSW9NBLCSN1F_OoXUSaQryWuuExpJeAdMoFA-9rmmfWwhmYXIEE41000.Pg7RpTIhXdbuTzS6C6hoTA.8BtbjT5qSz8LaqFH"  # Signature
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
captions = [
    "Big COCK.",
    "BIGGER COCK",
    "BIGGEST COCK."
]

# Sending Captions
seq = 4  # Starting sequence number
for caption in captions:
    response = send_caption(seq, caption)
    if response.status_code == 200:
        print(f"Caption {seq} sent successfully")
    else:
        print(f"Failed to send caption {seq}: {response.content}")
    seq += 1
    time.sleep(1)  # Pause for a second between captions
