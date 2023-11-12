# Steve

Steve AI is a tool that leverages multilingual translation and generative AI to help individuals communicate better. Some of the libraries used includes the Zoom API, Whisper, GPT-4 and Ngrok.


## To run the demo

First, start node-media-server on your local computer:

`npm i node-media-server -g && node-media-server`

Then, add the following code to the  `Ngrok` configuration file:

```
tunnels:
  rtmp:
    proto: tcp
    addr: 1935
  web:
    proto: http
    addr: 8000
```

and start both tunnels by running:

`ngrok start --all`

Next, follow this [Zoom instruction](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0060368) and copy/paste the API token appropiately in `caption_upload.py`, and run.

For multilanguge translation press `t` while speaking, and for definition press `d` while speaking. The procesed caption will be automatically uploaded to the current meeting session.
