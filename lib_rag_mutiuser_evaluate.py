import threading
import time
import random
import uuid
import sys

# 假設您的RAG程式碼儲存在 'your_rag_bot.py' 文件中
# 請確保這個文件在同一個目錄下，或者在Python的導入路徑中
try:
    from lib_rag import test as rag_chatbot_test, create_memory, reset_user_memory
except ImportError:
    print("錯誤：無法導入 'your_rag_bot.py'。請確保您的RAG程式碼已保存為此文件名並在同一目錄下。")
    sys.exit(1)

# --- 測試配置 ---
NUM_USERS = 5
QUESTIONS_PER_USER = 5
THINK_TIME_MIN = 1  # 每次提問後等待的最小時間（秒）
THINK_TIME_MAX = 3  # 每次提問後等待的最大時間（秒）
TIMEOUT_PER_QUESTION = 60 # 每個問題的最大處理時間（秒），防止無限等待

# 測試問題列表（中英文混合，模擬真實場景）
TEST_QUESTIONS = [
    "請問圖書館的開放時間？",
    "How do I borrow a book?",
    "我可以借幾本書?",
    "飲水機在哪?",
    "借書過期了怎辦?", # 模擬圖書查詢
    "請問可以使用館內電腦嗎？",
    "Can I print documents here?",
    "可以改暫停借書權利不要罰款嗎?",
    "如果查不到我要的書怎麼辦?", # 模擬圖書查詢
    "圖書館有提供論文諮詢服務嗎？",
    "我有幾位同學想討論事情有空間可以借嗎?",
    "請問有影印機嗎？",
    "聽說圖書館可以打電動? 真的嗎?",
    "圖書館有自修室嗎？",
    "Can I eat in the library?" # 模擬圖書查詢
]

# 用於收集測試結果
results = []
lock = threading.Lock() # 用於保護results列表的寫入

def user_task(user_id, chat_id):
    """
    模擬單一使用者的行為：發送多個問題並記錄結果。
    """
    print(f"[User {user_id}] ({chat_id}) 開始會話。")
    create_memory(chat_id) # 為新會話創建記憶體

    for i in range(QUESTIONS_PER_USER):
        question = random.choice(TEST_QUESTIONS)
        start_time = time.time()
        response = None
        error_message = None

        try:
            print(f"[User {user_id}] ({chat_id}) 提問 ({i+1}/{QUESTIONS_PER_USER}): '{question[:30]}...'")
            # 調用您的RAG聊天機器人函數
            response = rag_chatbot_test(chat_id, question)
            end_time = time.time()
            latency = end_time - start_time
            status = "成功"
            print(f"[User {user_id}] ({chat_id}) 回應 ({i+1}/{QUESTIONS_PER_USER}) 延遲: {latency:.2f} 秒")

            # 檢查回應是否為錯誤訊息，如果您的rag_chatbot_test在錯誤時會返回特定的錯誤字串
            if "錯誤" in response or "error" in response.lower():
                status = "失敗"
                error_message = response # 將錯誤訊息記錄下來
                print(f"[User {user_id}] ({chat_id}) 收到潛在錯誤回覆: {response[:50]}...") # 只印出部分
            elif not response: # 如果回覆為空
                status = "失敗"
                error_message = "回覆為空"
                print(f"[User {user_id}] ({chat_id}) 收到空回覆。")

        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            status = "失敗"
            error_message = str(e)
            print(f"[User {user_id}] ({chat_id}) 提問失敗 ({i+1}/{QUESTIONS_PER_USER}): {e}")

        # 將結果安全地添加到共享列表
        with lock:
            results.append({
                "user_id": user_id,
                "chat_id": chat_id,
                "question": question,
                "response": response,
                "latency": latency,
                "status": status,
                "error": error_message
            })

        # 模擬使用者思考時間
        if i < QUESTIONS_PER_USER - 1:
            think_time = random.uniform(THINK_TIME_MIN, THINK_TIME_MAX)
            time.sleep(think_time)

    print(f"[User {user_id}] ({chat_id}) 會話結束。")
    # 可以選擇在這裡呼叫 reset_user_memory(chat_id) 來明確清除該使用者的記憶體
    # 如果您的應用程式有自動清除機制（如 timeout），則不一定需要。
    # 為模擬長時間併發，通常不會在每個會話結束後立即清除。

def run_stress_test():
    """
    啟動所有使用者線程並管理測試報告。
    """
    print("--- 壓力測試開始 ---")
    print(f"模擬使用者數: {NUM_USERS}")
    print(f"每個使用者提問次數: {QUESTIONS_PER_USER}")
    print(f"總提問次數: {NUM_USERS * QUESTIONS_PER_USER}")
    print(f"提問間隔時間: {THINK_TIME_MIN}-{THINK_TIME_MAX} 秒\n")

    threads = []
    start_test_time = time.time()

    # 創建並啟動使用者線程
    for i in range(NUM_USERS):
        # 為每個使用者生成唯一的chat_id
        chat_id = str(uuid.uuid4())
        thread = threading.Thread(target=user_task, args=(i + 1, chat_id))
        threads.append(thread)
        thread.start()
        # 錯開啟動時間，避免所有請求同時打到服務器，模擬真實世界的併發
        time.sleep(0.1)

    # 等待所有線程完成
    for thread in threads:
        thread.join(timeout=TIMEOUT_PER_QUESTION * QUESTIONS_PER_USER * 2) # 給予足夠的等待時間

    end_test_time = time.time()
    total_test_duration = end_test_time - start_test_time

    print("\n--- 壓力測試結束 ---")
    print(f"總測試持續時間: {total_test_duration:.2f} 秒\n")

    # --- 生成報告 ---
    print("--- 測試報告 ---")
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r["status"] == "成功")
    failed_requests = total_requests - successful_requests

    print(f"總請求數: {total_requests}")
    print(f"成功請求數: {successful_requests}")
    print(f"失敗請求數: {failed_requests}")

    if total_requests > 0:
        success_rate = (successful_requests / total_requests) * 100
        print(f"成功率: {success_rate:.2f}%")

        latencies = [r["latency"] for r in results if r["status"] == "成功"]
        if latencies:
            min_latency = min(latencies)
            max_latency = max(latencies)
            avg_latency = sum(latencies) / len(latencies)
            print(f"成功請求最小延遲: {min_latency:.2f} 秒")
            print(f"成功請求最大延遲: {max_latency:.2f} 秒")
            print(f"成功請求平均延遲: {avg_latency:.2f} 秒")
        else:
            print("無成功請求可計算延遲。")
    else:
        print("沒有執行任何請求。")

    if failed_requests > 0:
        print("\n--- 失敗請求詳情 ---")
        for r in results:
            if r["status"] == "失敗":
                print(f"使用者 {r['user_id']} (chat_id: {r['chat_id'][:8]}...):")
                print(f"  問題: '{r['question'][:50]}...'")
                print(f"  延遲: {r['latency']:.2f} 秒")
                print(f"  錯誤: {r['error']}")
                if r['response']:
                    print(f"  回應: {r['response'][:50]}...") # 印出部分錯誤回應
                print("-" * 20)

    print("\n--- 測試報告結束 ---")

if __name__ == "__main__":
    run_stress_test()