import os
import tempfile
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, FileMessage, TextSendMessage
)
from PyPDF2 import PdfReader
import requests
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# 初始化 Flask
app = Flask(__name__)
load_dotenv()

# Line Bot 配置
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# Azure 翻譯配置
translator_credential = AzureKeyCredential(os.getenv('API_KEY'))
translator_client = TextTranslationClient(
    endpoint=os.getenv('ENDPOINT'),
    credential=translator_credential,
    region=os.getenv('REGION')
)

# 暫存 PDF 文件的問題上下文
user_context = {}

def translate_text(text, target_language="zh-Hant"):
    try:
        result = translator_client.translate(
            body=[text],
            to_language=[target_language]
        )
        return result[0].translations[0].text
    except Exception as e:
        return f"翻譯錯誤: {str(e)}"

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_pdf_qa_chain(pdf_path):
    # 使用 LangChain 載入 PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # 創建向量儲存
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 創建對話鏈
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    return qa_chain

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    
    return 'OK'

@handler.add(MessageEvent, message=FileMessage)
def handle_file_message(event):
    user_id = event.source.user_id
    message_id = event.message.id
    
    # 獲取文件
    message_content = line_bot_api.get_message_content(message_id)
    file_extension = event.message.file_name.split('.')[-1].lower()
    
    if file_extension != 'pdf':
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請上傳 PDF 文件！")
        )
        return
    
    # 儲存臨時文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(message_content.content)
        temp_file_path = temp_file.name
    
    # 儲存上下文
    user_context[user_id] = {
        'pdf_path': temp_file_path,
        'qa_chain': create_pdf_qa_chain(temp_file_path)
    }
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(
            text="PDF 已接收！請選擇操作：\n1. 翻譯全文至繁體中文\n2. 提問關於PDF內容的問題"
        )
    )

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_id = event.source.user_id
    text = event.message.text.strip()
    
    if user_id not in user_context:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請先上傳 PDF 文件！")
        )
        return
    
    pdf_path = user_context[user_id]['pdf_path']
    
    if text == "1":
        # 翻譯全文
        pdf_text = extract_pdf_text(pdf_path)
        translated_text = translate_text(pdf_text)
        
        # 由於 Line 訊息長度限制，取前1000字
        response = translated_text[:1000]
        if len(translated_text) > 1000:
            response += "...（內容過長，已截斷）"
            
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response)
        )
    
    elif text.startswith("2"):
        # 進入問答模式，等待具體問題
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請輸入您關於 PDF 內容的問題！")
        )
    
    else:
        # 處理問答
        qa_chain = user_context[user_id]['qa_chain']
        result = qa_chain({
            "question": text,
            "chat_history": []
        })
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result['answer'])
        )

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)