from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    PostbackEvent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    PostbackAction
)
import os

#Azure Translation
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

app = Flask(__name__)

CHANNEL_ACCESS_TOKEN = os.getenv("CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("CHANNEL_SECRET")

linehandler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # parse webhook body
    try:
        linehandler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@linehandler.add(MessageEvent, message=TextMessageContent)      ##代碼轉換網址:https://learn.microsoft.com/zh-tw/azure/ai-services/translator/language-support
def handle_messsage(event):
    text = event.message.text
    quick_reply_items=[
        QuickReplyItem(
            action=PostbackAction(
                label="英文",
                data=f"lang=en&text={text}",
                display_text="英文"
            )
        ),
        QuickReplyItem(
            action=PostbackAction(
                label="日文",
                data=f"lang=ja&text={text}",
                display_text="日文"
            )
        ),
        QuickReplyItem(
            action=PostbackAction(
                label="繁體中文",
                data=f"lang=zh-Hant&text={text}",
                display_text="繁體中文"
            )
        ),
        QuickReplyItem(
            action=PostbackAction(
                label="文言文",
                data=f"lang=lzh&text={text}",
                display_text="文言文"
            )
        )
    ]
    reply_message(event, [TextMessage(
        text='請選擇要翻譯的語言:',
        quick_reply=QuickReply(
            items=quick_reply_items
        )
    )])

@linehandler.add(PostbackEvent)
def handle_postback(event):
    postback_data = event.postback.data
    params = {}
    for param in postback_data.split("&"):
        key, value = param.split("=")
        params[key] = value
    user_input = params.get("text")
    language = params.get("lang")
    result = azure_translate(user_input, language)
    reply_message(event, [TextMessage(text=result if result else "No translation available")])

# 回覆訊息
def reply_message(event, messages):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=messages
            )
        )
        
# 處理Azure翻譯
def azure_translate(user_input, to_language):
    if to_language == None:
        return "Please select a language"
    else:
        apikey = os.getenv("API_KEY")
        endpoint = os.getenv("ENDPOINT")
        region = os.getenv("REGION")
        credential = AzureKeyCredential(apikey)
        text_translator = TextTranslationClient(credential=credential, endpoint=endpoint, region=region)
        
        try:
            response = text_translator.translate(body=[user_input], to_language=[to_language])
            print(response)
            translation = response[0] if response else None
            if translation:
                detected_language = translation.detected_language
                result = ''
                if detected_language:
                    print(f"偵測到輸入的語言: {detected_language.language} 信心分數: {detected_language.score}")
                for translated_text in translation.translations:
                    result += f"翻譯成: '{translated_text.to}'\n結果: '{translated_text.text}'"
                return result

        except HttpResponseError as exception:
            if exception.error is not None:
                print(f"Error Code: {exception.error.code}")
                print(f"Message: {exception.error.message}")

if __name__ == "__main__":
    app.run()