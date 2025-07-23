from langchain_community.vectorstores import Chroma #把文件變成向量(ex:HuggingFace)後儲存起來提供查詢
from langchain.text_splitter import RecursiveCharacterTextSplitter #將大段文字分割成更小的區塊
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate #用於傳送給語言模型的提示（prompt）結構和內容
from langchain.chains.combine_documents import create_stuff_documents_chain #用於將檢索到的文件與提示結合，然後輸入到語言模型中。 ex: lib_QA.txt
from langdetect import detect #輸入語言檢測
from langchain_huggingface import HuggingFaceEmbeddings #利用 HuggingFace Hub 上預訓練的嵌入模型（Embedding Models）來將文字轉換成數值向量。
from langchain_ollama import OllamaLLM #調用Ollama
from keyword_search import is_book_query, book_search_chi, book_search_eng #引入館藏查詢爬蟲程式
import time #引入時間偵測
from langchain.memory import ConversationBufferMemory #讓 AI 記住對話 啟用 Session控制對話時間 
from langchain_core.runnables import RunnableSequence #將多個 Runnable 組件串聯起來，形成一個線性的執行流程，就像一個管道 (pipeline)。 
from operator import itemgetter # Python 取得資料結構中的特定欄位或索引值 ex:提取用戶提問的輸入值。 

llm = OllamaLLM(model="gemma3:12b", base_url="http://127.0.0.1:11434")

embed_model = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large"
)


def load_text_from_file(file_path):
    """Loads text content from a specified file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}")
        return "" 
    except Exception as e:
        print(f"讀取檔案時發生錯誤 {file_path}: {e}")
        return ""

# 加載檔案內容
text = load_text_from_file('lib_QA.txt')

# 文本分割   以每1300字元作切割. 並保留上下文300個字元的重疊。 有了重疊，即使重要的資訊被分割了，也能在相鄰的塊中找到完整的上下文
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=300)
chunks = text_splitter.split_text(text)

# 建立向量資料庫
vector_store = Chroma.from_texts(chunks, embed_model)
retriever = vector_store.as_retriever()

# 中英文 prompt 模板    "根據對話歷史和提供的上下文，回答最後一個問題。 " 讓AI能夠理解問題的上下文，做出更連貫和相關的回應。例如，如果之前使用者問了「圖書館幾點開？」然後接著問「那週末呢？」，AI 就能根據對話歷史知道「那週末呢」是指週末的開放時間。
prompt_zh_template = """
你是一位親切又專業的圖書館服務人員，請務必使用繁體中文回答以下問題。非圖書館問題請勿回答。
以下是目前的對話歷史：
{chat_history}
根據對話歷史和提供的上下文，回答最後一個問題。 
<context>{context}</context>
Question:{input}
"""
prompt_zh = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_zh_template),
    HumanMessagePromptTemplate.from_template("{input}")
])

prompt_en_template = """
You are a professional librarian. You MUST answer the following question in English ONLY. Do not answer non-library related questions.
Here is the current conversation history:
{chat_history}
Based on the conversation history and the provided context, answer the last question.
<context>{context}</context>
Question:{input}
"""
prompt_en = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_en_template),
    HumanMessagePromptTemplate.from_template("{input}")
])
 
# 一定時間未對話即釋放GPU資源
last_active_time = {}
user_memories = {} 
SESSION_TIMEOUT = 15 * 60 # Session timeout in seconds (15 minutes)

def create_memory(chat_id):
    """Initializes memory and sets active time for a new session."""
    if chat_id not in user_memories:
        user_memories[chat_id] = ConversationBufferMemory(return_messages=True)
        print(f"建立 chat_id: {chat_id} 的新記憶體。")
    last_active_time[chat_id] = time.time()

def clear_memory(chat_id):
    """Clears memory and removes active time for a specific session."""
    if chat_id in user_memories:
        del user_memories[chat_id]
        print(f"清除 chat_id: {chat_id} 的記憶體。")
    if chat_id in last_active_time:
        del last_active_time[chat_id]
        print(f"移除 chat_id: {chat_id} 的活動時間。")

def reset_user_memory(chat_id):
    """Resets the memory for a specific user, called from app.py."""
    clear_memory(chat_id)


def get_document_chain(lang):
    """Returns the document chain based on the specified language."""
    if lang == 'en':
        prompt = prompt_en
    else:
        prompt = prompt_zh
    return create_stuff_documents_chain(llm, prompt)



def get_retrieval_chain(chat_id, input_text):
    """Builds and returns the retrieval chain and memory for a given chat_id."""
    memory = user_memories.get(chat_id)
    if not memory:
        print(f"警告: chat_id {chat_id} 的記憶體未找到，正在重新建立。")
        create_memory(chat_id)
        memory = user_memories[chat_id]

    chat_history = memory.load_memory_variables({})['history']


    lang = detect(input_text)
    document_chain = get_document_chain(lang)

    retrieval_chain = RunnableSequence(
        {
            "context": itemgetter("input") | retriever,
            "input": itemgetter("input"), 
            "chat_history": lambda x: chat_history
        } | document_chain
    )

    return retrieval_chain, memory 



def test(chat_id, input_text):
    """Main function to process user input and return a response."""
    current_time = time.time()

    # # 檢查 Session 是否過期
    # if chat_id not in last_active_time or current_time - last_active_time[chat_id] > SESSION_TIMEOUT:
    #     clear_memory(chat_id)
    #     return "很抱歉會話時間已過期，請重新整理頁面再對話。 Sorry, the session has expired. Please refresh the page and try again."

    # # 清理過期的其他 Session
    # inactive_sessions = [
    #     cid for cid, last_time in list(last_active_time.items())
    #     if cid != chat_id and current_time - last_time > SESSION_TIMEOUT
    # ]
    # for inactive_chat_id in inactive_sessions:
    #     clear_memory(inactive_chat_id)  
    #     print(f"釋放 chat_id: {inactive_chat_id} 的記憶體，閒置時間超過 {SESSION_TIMEOUT // 60} 分鐘。")


    book_title = is_book_query(input_text)
    if book_title: #如果 AI 成功回答了問題，就把「使用者問了什麼」和「AI 回答了什麼」這對問答記錄下來，添加到對話歷史中。
        lang = detect(input_text)  
        if lang == 'en':
            result = book_search_eng(book_title)
        else:
            result = book_search_chi(book_title)
    else:  #如果 AI 沒有成功生成回答，就發出一個警告。同時，它也避免將一個空的或無效的回答保存到對話記憶中，保持記憶的清潔和有效性。
        try:
            retrieval_chain, memory = get_retrieval_chain(chat_id, input_text)
            response = retrieval_chain.invoke({
                "input": input_text,
            })
            result = response  

            if result:
                 memory.save_context({"input": input_text}, {"output": str(result)})
            else:
                 print(f"警告: 從 RAG Chain 收到的結果為空或 None for chat_id {chat_id}")


        except Exception as e:
            print(f"Error during RAG chain invocation for chat_id {chat_id}: {e}")
            result = "處理您的請求時發生錯誤，請稍後再試。"


    # 更新活動時間
    last_active_time[chat_id] = time.time()
    return result

