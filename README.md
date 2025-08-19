需要另外安裝的開源套件有:
Python 建議 3.10.8 版本
ollama 安裝網頁 https://ollama.com/download
gemma3:12b 可在本地 Ollama 中下載
pip install langchain-ollama //使用 langchain_ollama 來呼叫本地 Ollama 的 Gemma 模型
pip install flask  //用於建立 Web 應用伺服器
pip install pymysql //與 MySQL 資料庫進行連線與操作
pip install langchain //用於 RAG（Retrieval-Augmented Generation）運作
pip install langdetect //用於偵測使用者輸入的語言
pip install -U langchain-ollama //整合 Ollama 模型與 LangChain 用於 LLM 呼叫
pip install chromadb //安裝Chroma 向量庫
pip install langchain langchain-community //用於構建對話記憶、prompt chaining、向量檢索等功能
pip install huggingface-hub transformers //使用 HuggingFace 提供的嵌入模型 (HuggingFaceEmbeddings)
pip install sentence-transformers //使用HuggingFace 上的 thenlper/gte-large 模型嵌入書籍內容

MySQL表單建立語法:

CREATE TABLE `qa_table` (
  `id` int NOT NULL AUTO_INCREMENT,
  `session_id` varchar(80) DEFAULT NULL,
  `lib` varchar(45) DEFAULT NULL,
  `question` varchar(200) DEFAULT NULL,
  `answer` varchar(500) DEFAULT NULL,
  `thumbs` varchar(10) DEFAULT NULL,
  `day_time` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=156 DEFAULT CHARSET=utf8mb3;
