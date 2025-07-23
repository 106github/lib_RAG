from bs4 import BeautifulSoup #Python 中用於解析 HTML 和 XML 文件的函式庫
import requests #當你在瀏覽器中輸入一個網址（URL）並按下 Enter 鍵時，瀏覽器會向該網址的伺服器發送一個 HTTP 請求，然後伺服器會返回網頁的內容。requests 函式庫就提供了在 Python 程式中執行這種操作的能力。
import re # 引入的是 Python 的內建模組，用於處理正規表達式 (Regular Expressions)。
import urllib #將使用者提供的書籍查詢關鍵字 (query) 進行 URL 編碼。
from bs4 import Tag #在 Beautiful Soup 解析 HTML文件後，它會將文件中的每個 HTML/XML 元素（例如 <div>、<p>、<a>、<td> 等）都表示為一個 Tag 物件。
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate  #引入了 LangChain 框架中用於構造與大型語言模型 (LLM) 互動提示 (Prompt) 的關鍵類別。
from langchain_community.llms import Ollama
 
# 檢查是否是書籍查詢，並返回查詢網址
def is_book_query(input_text):
    print(f"is_book_query 接收到的輸入: {input_text}")

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""你是一個圖書館助理。你的任務是判斷用戶的輸入是否在詢問尋找特定的書籍。如果是，請提取書名或相關關鍵字。如果不是，請回答 "否"。"""),
        HumanMessagePromptTemplate.from_template("{user_input}")
    ])

    try:
        llm = Ollama(model="gemma3:12b", base_url="http://127.0.0.1:11434")
        llm_chain = prompt_template | llm
        response_content = llm_chain.invoke({"user_input": input_text})
        print(f"is_book_query 的輸出: {response_content}")

        if "否" in response_content.lower():
            return False
        else:
            book_title = None
            # 嘗試匹配不同的前綴
            match = re.search(r"(?:書名是|書名或關鍵字是|書名或關鍵字|書名)：\s*(.+)", response_content)
            if match:
                book_title = match.group(1).strip()
            else:
                # 如果沒有匹配到前綴，則嘗試直接提取 "我要找" 和 "這本書" 之間的內容
                match_explicit = re.search(r"我要找(.+)這本書", input_text)
                if match_explicit:
                    book_title = match_explicit.group(1).strip()
                else:
                    book_title = response_content.strip()

            # 移除書名中的《》符號
            if book_title:
                book_title = book_title.replace("《", "").replace("》", "")
                return book_title
            else:
                return False # 如果 book_title 為 None，則返回 False

    except Exception as e:
        print(f"Error in is_book_query: {e}")
        return False


def book_search_chi(query):
    query = urllib.parse.quote(query, encoding='utf-8')
    url = f"https://xxx/F?func=find-b&find_code=WRD&adjacent=Y&local_base=FLY03&request={query}"
    print(f"正在查詢的 URL: {url}") # 加入這行
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "無法訪問書籍資料庫，請稍後再試。"

    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    script_content = soup.find('script')
    if script_content:
        script_string = script_content.string
        new_url = re.findall(r'https?://[^\s;]+', script_string, re.DOTALL)
        if new_url:
            new_url = new_url[0].replace("'", "")
            match = re.search(r"&url=(.+)", new_url)
            if match:
                extracted_url = match.group(1)
                resp = requests.get(extracted_url, headers=headers)
                if resp.status_code != 200:
                    return "無法獲取書籍詳情，請稍後再試。"

                resp.encoding = 'utf-8'
                soup = BeautifulSoup(resp.text, 'html.parser')

                text3_cell = soup.find('td', class_='text3')
                if text3_cell:
                    cleaned_text = ' '.join(text3_cell.text.split())
                    match = re.search(r'of (\d+) 筆', cleaned_text)  # 共幾本書

                    td_elements = soup.find_all('td', class_='td1', width='30%') # 擷取前三本書名

                    while len(td_elements) < 3: # 當比數小於3筆時用空元素取代
                        empty_tag = Tag(soup, name="td")  # 創建一個空的 td 元素
                        td_elements.append(empty_tag)

                    book_titles = "《"+ td_elements[0].get_text(strip=True) + "》《" + td_elements[1].get_text(strip=True) + "》《" + td_elements[2].get_text(strip=True)+"》"
                    text_content = book_titles.rstrip("/")

                    if match:
                        total = match.group(1)
                        return f"找到 {total} 筆相關書籍. 例如{text_content}... 或者您可以到館藏查詢頁面去搜尋：{query}"
                    else:
                        return "未能在資料庫中找到相關書籍的數量。您可以改用 找xxx的書，讓我再幫忙找一次。"
                else:
                    return "未能找到書籍。您可以改用其他關鍵字讓我再找一次，或者直接去館藏查詢頁面去搜尋。"
            else:
                return "無法提取有效的查詢結果 URL"
        else:
            return "未能找到有效的腳本內容，可能需要調整網頁解析。"
    else:
        return "未能找到有效的腳本內容，可能需要調整網頁解析。"



def book_search_eng(query):
    query = urllib.parse.quote(query, encoding='utf-8')
    url = f"https://xxx/F?func=find-b&find_code=WRD&adjacent=Y&local_base=FLY03&request={query}"  # 這裡有URL構建
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Unable to access the book database. Please try again later."

    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')

    script_content = soup.find('script')
    if script_content:
        script_string = script_content.string
        new_url = re.findall(r'https?://[^\s;]+', script_string, re.DOTALL)
        if new_url:
            new_url = new_url[0].replace("'", "")
            match = re.search(r"&url=(.+)", new_url)
            if match:
                extracted_url = match.group(1)
                resp = requests.get(extracted_url, headers=headers)
                if resp.status_code != 200:
                    return "Unable to access the book database. Please try again later."

                resp.encoding = 'utf-8'
                soup = BeautifulSoup(resp.text, 'html.parser')

                text3_cell = soup.find('td', class_='text3')
                if text3_cell:
                    cleaned_text = ' '.join(text3_cell.text.split())
                    match = re.search(r'of (\d+) 筆', cleaned_text)  # 共幾本書

                    td_elements = soup.find_all('td', class_='td1', width='30%') # 擷取前三本書名

                    while len(td_elements) < 3: # 當比數小於3筆時用空元素取代
                        empty_tag = Tag(soup, name="td")  # 創建一個空的 td 元素
                        td_elements.append(empty_tag)

                    book_titles = "《"+ td_elements[0].get_text(strip=True) + "》《" + td_elements[1].get_text(strip=True) + "》《" + td_elements[2].get_text(strip=True)+"》"
                    text_content = book_titles.rstrip("/")

                    if match:
                        total = match.group(1)
                        return f"Found {total} related books. For example,{text_content}... or you can visit the catalog search page to search：{query}"
                    else:
                        return f"Unable to find the number of related books in the database. You can try using 'find the book by xxx', and I can help you search again."
                else:
                    return f"Unable to find the number of related books in the database. You can try using 'find the book by xxx', and I can help you search again."
            else:
                return "Unable to retrieve a valid query result URL"
        else:
            return "Unable to find valid script content. Adjustments to the webpage parsing may be needed."
    else:
        return "Unable to find valid script content. Adjustments to the webpage parsing may be needed."
