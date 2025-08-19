需要另外安裝的開源套件有:
Python 建議 3.10.8 版本<br>
ollama 安裝網頁 https://ollama.com/download<br>
gemma3:12b 可在本地 Ollama 中下載<br>
pip install langchain-ollama //使用 langchain_ollama 來呼叫本地 Ollama 的 Gemma 模型<br>
pip install flask  //用於建立 Web 應用伺服器<br>
pip install pymysql //與 MySQL 資料庫進行連線與操作<br>
pip install langchain //用於 RAG（Retrieval-Augmented Generation）運作<br>
pip install langdetect //用於偵測使用者輸入的語言<br>
pip install -U langchain-ollama //整合 Ollama 模型與 LangChain 用於 LLM 呼叫<br>
pip install chromadb //安裝Chroma 向量庫<br>
pip install langchain langchain-community //用於構建對話記憶、prompt chaining、向量檢索等功能<br>
pip install huggingface-hub transformers //使用 HuggingFace 提供的嵌入模型 (HuggingFaceEmbeddings)<br>
pip install sentence-transformers //使用HuggingFace 上的 thenlper/gte-large 模型嵌入書籍內容<br>
<br>
MySQL表單建立語法:<br>

CREATE TABLE `qa_table` (<br>
  `id` int NOT NULL AUTO_INCREMENT,<br>
  `session_id` varchar(80) DEFAULT NULL,<br>
  `lib` varchar(45) DEFAULT NULL,<br>
  `question` varchar(200) DEFAULT NULL,<br>
  `answer` varchar(500) DEFAULT NULL,<br>
  `thumbs` varchar(10) DEFAULT NULL,<br>
  `day_time` varchar(20) DEFAULT NULL,<br>
  PRIMARY KEY (`id`),<br>
  UNIQUE KEY `id_UNIQUE` (`id`)<br>
) ENGINE=InnoDB AUTO_INCREMENT=156 DEFAULT CHARSET=utf8mb3;
