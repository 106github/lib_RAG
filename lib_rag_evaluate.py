from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langdetect import detect
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from keyword_search import is_book_query, book_search_chi, book_search_eng
import time
import logging

# 設定 logging 等級，避免 langchain 輸出過多的資訊
logging.getLogger("langchain_retrievers.base").setLevel(logging.ERROR)
logging.getLogger("langchain.chains.retrieval_qa.base").setLevel(logging.ERROR)

# 初始化 Ollama LLM
llm = OllamaLLM(model="gemma3:12b", base_url="http://127.0.0.1:11434")

# 正確 HuggingFace Embedding 版本
embed_model = HuggingFaceEmbeddings(
    model_name="thenlper/gte-large"
)

# 讀取外部 txt 檔
def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 加載 lib_QA.txt 內容
text = load_text_from_file('lib_QA.txt')

# 文本分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=200)
chunks = text_splitter.split_text(text)

# 建立向量資料庫
vector_store = Chroma.from_texts(chunks, embed_model)
retriever = vector_store.as_retriever()

# 中英文 prompt 模板 (包含更明確的語言指示)
prompt_zh_template = """
你是一位親切又專業的圖書館服務人員，請務必使用繁體中文回答以下問題。非圖書館問題請勿回答。
<context>{context}</context>
Question:{input}
"""
prompt_zh = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_zh_template),
    HumanMessagePromptTemplate.from_template("{input}")
])

prompt_en_template = """
You are a professional librarian. You MUST answer the following question in English ONLY. Do not answer non-library related questions.
<context>{context}</context>
Question:{input}
"""
prompt_en = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(prompt_en_template),
    HumanMessagePromptTemplate.from_template("{input}")
])

# 記憶體管理
last_active_time = {}
SESSION_TIMEOUT = 15 * 60

def create_memory(chat_id):
    last_active_time[chat_id] = time.time()

# 主測試方法 (保持與你提供的程式碼一致)
def test(chat_id, input_text):
    current_time = time.time()

    # 檢查 Session 是否過期
    if chat_id not in last_active_time or current_time - last_active_time[chat_id] > SESSION_TIMEOUT:
        if chat_id in last_active_time:
            del last_active_time[chat_id]
        return "很抱歉會話時間已過期，請重新整理頁面再對話。 Sorry, the session has expired. Please refresh the page and try again."

    # 清理過期的其他 Session
    inactive_sessions = [
        cid for cid, last_time in last_active_time.items()
        if cid != chat_id and current_time - last_time > SESSION_TIMEOUT
    ]
    for inactive_chat_id in inactive_sessions:
        del last_active_time[inactive_chat_id]
        print(f"釋放 chat_id: {inactive_chat_id} 的記憶體，閒置時間超過 {SESSION_TIMEOUT // 60} 分鐘。")

    book_title = is_book_query(input_text)
    if book_title:
        lang = detect(input_text)
        if lang == 'en':
            result = book_search_eng(book_title)
        else:
            result = book_search_chi(book_title)
    else:
        lang = detect(input_text)
        if lang == 'en':
            prompt = prompt_en
        else:
            prompt = prompt_zh

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        response = retrieval_chain.invoke({
            "input": input_text,
        })
        result = response['answer']

    # 更新活動時間
    last_active_time[chat_id] = time.time()

    return result

# 自動測試程式 (修改後，加入匯出到檔案的功能)
def auto_test(test_file_path="lib_rag_evaluate_QA.txt", output_file_path="evaluate_QA_report.txt"):
    qa_list = []
    try:
        with open(test_file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]

        chat_id = "auto_tester"  # 使用固定的 chat_id 進行測試，避免 session 過期影響
        create_memory(chat_id)

        for question in questions:
            answer = test(chat_id, question)
            qa_list.append({"question": question, "answer": answer})
            print(f"問題：{question}")
            print(f"回答：{answer}\n")

        # 匯出到 txt 檔案
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write("--- 提問與回答清單 ---\n")
            for qa in qa_list:
                outfile.write(f"提問：{qa['question']}\n")
                outfile.write(f"回答：{qa['answer']}\n")
                outfile.write("-" * 20 + "\n")

        print(f"\n提問與回答已匯出至 {output_file_path}")

    except FileNotFoundError:
        print(f"錯誤：找不到測試檔案 {test_file_path}")
    except Exception as e:
        print(f"發生錯誤：{e}")
    return qa_list

if __name__ == "__main__":
    results = auto_test()