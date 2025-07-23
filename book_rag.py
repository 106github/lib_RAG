import json
import re
import torch
import time
from langchain.memory import ConversationSummaryBufferMemory
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# 根據 LangChain 的提示，調整 HuggingFaceEmbeddings 的導入路徑
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 模型和嵌入設定 ---
llama_model = "gemma3:12b"
llm = OllamaLLM(model=llama_model, base_url="http://127.0.0.1:11434")

embedding_model = HuggingFaceEmbeddings(model_name='thenlper/gte-large') # 適合更多複雜文本的模型，擁有更多的參數。

# --- 書籍資料載入與處理 ---
try:
    with open('./book.json', 'r', encoding='utf-8') as f:
        books = json.load(f)
except FileNotFoundError:
    print("錯誤：找不到 ./book.json 文件。請確保文件存在。")
    books = []

book_embeddings = []
book_texts = []
if books:
    for book in books:
        book_info = {key: book.get(key, '') for key in ['書名', '作者', '分類', '出版社', '簡介']}
        content = f"{book_info['書名']}。{book_info['作者']}。{book_info['分類']}。{book_info['出版社']}。{book_info['簡介']}"
        book_texts.append(content)
    embeddings = embedding_model.embed_documents(book_texts)

    for i, book in enumerate(books):
        if i < len(embeddings):
            book_embeddings.append({"book": book, "embedding": torch.tensor(embeddings[i])})
        else:
            print(f"警告：書籍 '{book.get('書名', '未知')}' 未能成功編碼，跳過。")

available_titles = "\n".join([f"- {book.get('書名', '未知')} {book.get('作者', '未知')}" for book in books])

# --- 會話記憶體管理 ---
test_all = {}
last_active_time = {}
# 設定逾時時間為 15 分鐘 (秒) 用戶無對話就釋放記憶體
SESSION_TIMEOUT = 15 * 60

def create_memory(chat_id):
    """為新的 chat_id 創建記憶體和初始化狀態"""
    if chat_id not in test_all:
        print(f"創建會話記憶體，chat_id: {chat_id}")
        test_all[chat_id] = {
            "memory": ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=1000,
                memory_key="chat_history"
            ),
            "last_topic": None, # 代表上一個模型回答的書名
            "current_book_topic": None # 代表目前對話正在圍繞的書名 (更具焦點性)
        }
    last_active_time[chat_id] = time.time()

def remove_memory(chat_id):
    """根據 chat_id 移除記憶體和狀態"""
    if chat_id in test_all:
        print(f"移除會話記憶體，chat_id: {chat_id}")
        del test_all[chat_id]
    if chat_id in last_active_time:
        del last_active_time[chat_id]

def reset_user_memory(chat_id):
    """重置特定使用者的記憶體 (與 remove_memory 相同功能)"""
    remove_memory(chat_id)

def cleanup_inactive_sessions():
    """清理超時不活動的會話記憶體"""
    current_time = time.time()
    inactive_sessions = [
        chat_id for chat_id, last_time in last_active_time.items()
        if current_time - last_time > SESSION_TIMEOUT
    ]
    for chat_id in inactive_sessions:
        remove_memory(chat_id)
        print(f"會話 {chat_id} 因逾時而被清理。")

# --- 書籍相關工具函數 ---
def semantic_search(query, top_k=5):
    """根據查詢進行語義搜索並返回最相關的書籍"""
    if not book_embeddings:
        return []
    query_embedding = torch.tensor(embedding_model.embed_query(query))
    scores = []
    for book_data in book_embeddings:
        book_emb = book_data['embedding']
        score = torch.cosine_similarity(query_embedding, book_emb, dim=0).item()
        scores.append((book_data['book'], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k] if scores else []

def get_book_description_by_title(title):
    """根據書名獲取書籍簡介"""
    for book in books:
        if book.get('書名') == title:
            return book.get('簡介', '沒有相關簡介')
    return "沒有相關資訊"

def is_comment_sentence(input_text):
    """判斷輸入是否為評論或感想句，或是否適合推薦"""
    # 增加更多判斷詞彙，涵蓋「是否適合」這類追問
    return bool(re.search(r"(看起來|聽起來|應該是|描述的是|講的是|說的是|感覺像|心得|感想|如何|評價|看法|適合我|幫我判斷|怎麼樣)", input_text, re.IGNORECASE))

# --- LangChain 提示詞和鏈 ---
restricted_prompt = PromptTemplate(
    input_variables=["chat_history", "question", "book_list"],
    template="""
你是專業的書籍推薦與知識解說者，請用繁體中文親切且專業地回答。

📚【重要指示】：你必須嚴格遵守以下規則：
1.  當用戶詢問書單或要求推薦書籍時，你只能從以下提供的書單中回答或推薦，禁止推薦任何不在這個列表中的書籍，絕對不能虛構書名。
2.  當你推薦多本書籍時，請針對每一本書籍先提供書名和作者，然後是簡短的介紹，並在每本書的介紹結束後加上兩個換行符號<br><br>。
3.  如果用戶詢問的書籍不在這個列表中，請明確回答「抱歉我的書單中找不到」。
4.  對於非書籍相關的問題，你可以自由回答（例如：「什麼是ETF？」、「今天天氣如何？」），**但請務必留意對話歷史中的「焦點書籍」。當用戶再次提及代詞（如「這本」、「它」）時，即使中間穿插了其他話題，也請將問題關聯到最近一次討論的焦點書籍。**
5.  如果判斷不出用戶指的具體書籍，請禮貌地詢問用戶提供書名。

📚【可用書單】：
{book_list}

💬【對話紀錄】：
{chat_history}
User: {question}
"""
)
restricted_chain = LLMChain(llm=llm, prompt=restricted_prompt)  

comment_reply_prompt = PromptTemplate(
    input_variables=["comment", "book_intro", "book_title"],
    template="""
你是專業書籍解說者。
以下是讀者對《{book_title}》的評論或感想，請親切專業回應，可補充主題重點，但避免直接複製簡介。

讀者評論：{comment}
書本背景：{book_intro}

請開始回應：
"""
)
comment_reply_chain = LLMChain(llm=llm, prompt=comment_reply_prompt)

# --- 主對話處理函數 ---
def test(chat_id, input_text):
    """處理用戶的輸入，返回回答，並管理會話記憶體"""
    cleanup_inactive_sessions() # 清理過期會話

    if chat_id not in test_all:
        print(f"會話 {chat_id} 不存在或已過期，創建新的記憶體。")
        create_memory(chat_id)

    last_active_time[chat_id] = time.time()
    print(f"更新會話 {chat_id} 最後活動時間: {last_active_time[chat_id]}")

    memory = test_all[chat_id]["memory"]
    last_topic = test_all[chat_id]["last_topic"]
    current_book_topic = test_all[chat_id]["current_book_topic"]

    book_titles = [book.get('書名') for book in books if book.get('書名')]
    found_book_title_in_input = None

    # 1. 檢查輸入是否直接包含書名
    for title in book_titles:
        # 使用更精確的匹配，避免部分詞彙誤判 (例如"人生"可能匹配到多本書)
        if re.search(r'\b' + re.escape(title) + r'\b', input_text, re.IGNORECASE):
            found_book_title_in_input = title
            break

    # 如果輸入直接包含書名，則將其設定為當前書籍話題並直接返回簡介
    if found_book_title_in_input:
        print(f"輸入包含書名: {found_book_title_in_input}")
        test_all[chat_id]["current_book_topic"] = found_book_title_in_input
        test_all[chat_id]["last_topic"] = found_book_title_in_input # 直接查詢書名也視為上一個話題
        description = get_book_description_by_title(found_book_title_in_input)
        memory.save_context({"input": input_text}, {"output": description})
        return description

    # 2. 處理代詞和對焦點書籍的追問
    processed_input = input_text
    # 檢查是否包含常用的代詞 (他, 它, 這本等) 或常見的追問短語 (適合我, 怎麼樣等)
    is_pronoun_present = bool(re.search(r"\b(他|它|她|這個|這本|那個|該|此)\b", input_text, re.IGNORECASE))
    is_follow_up_question_on_book = is_comment_sentence(input_text) # 延用 is_comment_sentence 判斷是否適合這類問題

    # 如果有代詞或判斷是追問，且目前有焦點書籍，且輸入中沒有明確的書名
    if (is_pronoun_present or is_follow_up_question_on_book) and current_book_topic and not found_book_title_in_input:
        print(f"偵測到代詞或對焦點書籍 ({current_book_topic}) 的追問。")
        # 構造更明確的輸入給模型，將焦點書籍強行注入
        processed_input = f"請問您提及的『{current_book_topic}』，{input_text.replace('這本', '').replace('這個', '').replace('它', '').strip()}？"

        # 如果確實是針對焦點書籍的「適合與否」或「看法」追問，直接走 comment_reply_chain
        if is_follow_up_question_on_book:
            print(f"判斷為對焦點書籍 ({current_book_topic}) 的評論/感想/追問。直接導向 comment_reply_chain。")
            book_intro = get_book_description_by_title(current_book_topic)
            response = comment_reply_chain.run(
                comment=input_text, # 使用原始的用戶輸入作為評論
                book_intro=book_intro,
                book_title=current_book_topic
            )
            memory.save_context({"input": input_text}, {"output": response})
            # 在這裡，current_book_topic 保持不變，last_topic 也保持不變
            return response
        else:
            # 如果是其他類型的代詞使用，但不是 "適合與否" 的判斷，則仍然走 restricted_chain
            print(f"處理代詞，但不直接導向 comment_reply_chain。 processed_input: {processed_input}")


    # 3. 處理一般問題 (包括 RAG 鏈)
    # 如果上面沒有直接返回，則執行此處的邏輯
    print(f"準備使用 RAG 鏈處理輸入: {processed_input}")
    history = memory.chat_memory.messages
    chat_history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history]) if history else ""

    result = restricted_chain.run(
        chat_history=chat_history_text,
        question=processed_input,
        book_list=available_titles
    )
    memory.save_context({"input": input_text}, {"output": result})

    # --- 關鍵修改：從模型回答中尋找書名，更新 last_topic 和 current_book_topic ---
    found_topic_in_response = None
    # 嘗試用更彈性的方式尋找模型回答中的書名
    # 對模型回答和書名進行預處理，移除所有非字母數字的字元，並轉為小寫，以增加匹配的容錯率
    cleaned_result = re.sub(r'[^\w]', '', result).lower()

    for title in book_titles:
        cleaned_title = re.sub(r'[^\w]', '', title).lower()
        if cleaned_title and cleaned_title in cleaned_result:
            found_topic_in_response = title # 儲存原始書名，因為它是正確的名稱
            break

    if found_topic_in_response:
        test_all[chat_id]["last_topic"] = found_topic_in_response
        test_all[chat_id]["current_book_topic"] = found_topic_in_response
        print(f"從模型回答中找到話題，更新 last_topic 和 current_book_topic: {found_topic_in_response}")
    else:
        # 如果模型回答中沒有新的書名，current_book_topic 保持不變，last_topic 也不更新。
        # 這樣做的目的是當使用者從非書籍問題再回到「這本」書時，模型能記得是哪本。
        print(f"模型回答中未找到新書名。current_book_topic: {current_book_topic}, last_topic: {last_topic} 保持不變。")

    return result

# --- 範例使用 (您可以取消註解並執行來測試) ---
# if __name__ == "__main__":
#     test_chat_id = "user123"

#     print("--- 第一輪對話：推薦書籍 (理財) ---")
#     response1 = test(test_chat_id, "我是理財小白,你有推薦簡單的理財書嗎?")
#     print(f"Bot: {response1}")
#     print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}")
#     print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}")
#     print("-" * 30)
#     time.sleep(1) # 模擬延遲

#     print("--- 第二輪對話：詢問特定書籍詳情 (理財書) ---")
#     response2 = test(test_chat_id, "零基礎的佛系理財術：只要一招，安心穩穩賺他在寫什麼?")
#     print(f"Bot: {response2}")
#     print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}")
#     print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}")
#     print("-" * 30)
#     time.sleep(1)

#     print("--- 第三輪對話：非書籍問題 (股票) ---")
#     response3 = test(test_chat_id, "股票要怎麼投資啊?")
#     print(f"Bot: {response3}")
#     print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}")
#     print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}") # 應該還是理財書
#     print("-" * 30)
#     time.sleep(1)

#     print("--- 第四輪對話：使用代詞追問之前的書籍 (理財書 - 關鍵測試點 1) ---")
#     response4 = test(test_chat_id, "所以這本應該適合我吧?")
#     print(f"Bot: {response4}")
#     print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}")
#     print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}")
#     print("-" * 30)
#     time.sleep(1)

#     print("--- 第五輪對話：新的書籍推薦需求 (自然文學) ---")
#     response5 = test(test_chat_id, "有沒有跟自然有關但比較文學一點的書？")
#     print(f"Bot: {response5}")
#     print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}") # 期望是《遙遠的向日葵地》
#     print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}") # 期望是《遙遠的向日葵地》
#     print("-" * 30)
#     time.sleep(1)

#     print("--- 第六輪對話：使用代詞追問新的書籍 (自然文學 - 關鍵測試點 2) ---")
#     response6 = test(test_chat_id, "你認為這本適合我嗎?")
#     print(f"Bot: {response6}")
#     print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}") # 期望仍是《遙遠的向日葵地》
#     print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}")
#     print("-" * 30)
#     time.sleep(1)

#     print("--- 第七輪對話：詢問另一本書籍 ---")
#     response7 = test(test_chat_id, "《你要如何衡量你的人生？：哈佛商學院最重要的一堂課》這本書在講什麼?")
#     print(f"Bot: {response7}")
#     print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}")
#     print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}")
#     print("-" * 30)
#     time.sleep(1)

#     # 模擬逾時清理
#     print("\n--- 模擬逾時清理 ---")
#     time.sleep(SESSION_TIMEOUT + 5) # 等待超過逾時時間
#     response_after_timeout = test(test_chat_id, "我剛剛問到哪裡了？")
#     print(f"Bot (逾時後): {response_after_timeout}")
#     # 這裡的 current_book_topic 和 last_topic 應該會是 None
#     if test_chat_id in test_all:
#         print(f"Current Book Topic: {test_all[test_chat_id].get('current_book_topic')}")
#         print(f"Last Topic: {test_all[test_chat_id].get('last_topic')}")
#     else:
#         print(f"會話 {test_chat_id} 已被清理。")