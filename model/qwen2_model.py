
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Optional
# from fastapi import APIRouter, HTTPException, Header , Request
# from pydantic import BaseModel
# import asyncio
# from starlette.concurrency import run_in_threadpool
# from linebot import LineBotApi, WebhookHandler
# from linebot.exceptions import InvalidSignatureError
# from linebot.models import TextMessage, MessageEvent, TextSendMessage, StickerMessage, \
#     StickerSendMessage
    
import requests
import json
# from app.common.response.response_schema import ResponseModel, response_base
# from app.utils.server_info import server_info
# from app.common.log import log

# router = APIRouter(prefix="/lineoa")

url = "http://192.168.123.110:11434/api/generate"
# # line_bot_api = LineBotApi("fICknsI85JLgxQ7JQREb/RR5PWgG0hBvsU0G4Ps0jLIH3TnlmK3CYagpfwo9dlrucdYfYnr8cbhGVQbgyqo1iSd6akvRo3a6ETID38aQuupfZLAMXkp/49tbTUGGNKJwDcj8GNs1bAvBURJAXcLNBAdB04t89/1O/w1cDnyilFU=")
# # handler = WebhookHandler("bd97c1102e645c468fb29fc44a1e0f31")
# line_bot_api = LineBotApi("VbX/Yzr3+C/34kaE7RRC8Z1SEYgSv8N7P6KO/sflEDiFDGNNUw/c9l6KaNCXf7hltG4TXfP5Rg/tGbI9umiQS/rQA4iV0x4BoVWRqqL1OgMekWbVeFzMMyg64hMzNjMoM3DatVYUpGp97cdeKAE3sQdB04t89/1O/w1cDnyilFU=")
# handler = WebhookHandler("59da6fbca166f1cd17cb8505dff3267f")

# router = APIRouter(
#     prefix="/webhooks",
#     responses={404: {"description": "Not found"}},
# )

# # @router.get('/hello-test')
# # def hello_word_test():
# #     return {"hello" : "world"}

# @router.post('/message')
# async def hello_word(request: Request):
#     signature = request.headers['X-Line-Signature']
#     body = await request.body()
    
#     try:
#         handler.handle(body.decode('UTF-8'), signature)
#         pass
#     except InvalidSignatureError:
#         print("Invalid signature. Please check your channel access token/channel secret.")
#     return 'OK'


def call_llm(message: str):
    """
    Call ollama API
    """
    print(f"Call LLM with message: [{message}]")
    payload = json.dumps({
    # "model": "llama3",
    "model": "qwen2:72b",
    "prompt": message,
    "stream": False
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_json = json.loads(response.text)
    return response_json['response']

# @handler.add(MessageEvent, message=TextMessage)
# def handle_message(event):
#         asyncio.create_task(sendMessage(event, event.message.text))
            
# async def sendMessage(event,message):
#     # reply_message = call_llm(message)
#     reply_message = await run_in_threadpool(
#          call_llm,
#          message
#     )
#     if not reply_message:
#         return
#     reply_token = event.reply_token
#     if not reply_token:
#         print(f"Token not found")
#         return
#     print(f"Reply token: {reply_token}")
#     line_bot_api.reply_message(
#         reply_token=reply_token,
#         messages=TextSendMessage(text=reply_message))
    

message = "โครงสร้างเศรษฐกิจของจังหวัดเชียงรายมาจากการเกษตร"
reply = call_llm(message)
print(reply)