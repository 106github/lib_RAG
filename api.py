from flask import Flask, jsonify, request, session, render_template
# 您的RAG函式庫
from lib_rag import test as test_lib, create_memory, reset_user_memory
# from book_rag import test as test_lib, create_memory, reset_user_memory
# from mcut_rag import test as test_lib, create_memory, reset_user_memory
# from care_rag import test as test_lib, create_memory, reset_user_memory

from langchain.memory import ConversationBufferMemory
import datetime
import uuid
import pymysql
import os

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'hahaha')

# --- 資料庫設定 ---
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "passwd": "1234",
    "database": "lib_qa",
    "charset": "utf8"
}

@app.route('/')
def index():
    if 'chat_id' not in session:
        session['chat_id'] = generate_id()
        create_memory(session['chat_id'])
    return render_template('index.html', chat_id=session['chat_id'])

@app.route('/lib_test', methods=['POST'])
def lib_test():
    if not request.json or 'question' not in request.json:
        return jsonify({'error': '請輸入問題'}), 400

    my_question = request.json['question']
    chat_id = session.get('chat_id')

    if not chat_id:
        print("Warning: chat_id missing from session in /lib_test. Generating new one.")
        session['chat_id'] = generate_id()
        chat_id = session['chat_id']
        create_memory(chat_id)

    my_answer = test_lib(chat_id, my_question)
    
    db = None
    new_record_id = None
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        
        lib_value = "mcut"
        current_date = datetime.date.today()
        
        sql = "INSERT INTO qa_table(session_id, lib, question, answer, thumbs, day_time) VALUES(%s, %s, %s, %s, NULL, %s)"
        cursor.execute(sql, (chat_id, lib_value, my_question, my_answer, current_date))
        db.commit()
        
        # --- 修改重點 1: 取得剛剛新增紀錄的ID ---
        new_record_id = cursor.lastrowid
        print(f"Auto-recorded Q&A for chat_id: {chat_id} with record ID: {new_record_id}")
    
    except pymysql.Error as e:
        if db: db.rollback()
        print(f"Database error during auto-record in /lib_test: {e}")
    except Exception as e:
        if db: db.rollback()
        print(f"An unexpected error occurred during auto-record: {e}")
    finally:
        if db: db.close()

    # --- 修改重點 2: 將紀錄ID回傳給前端 ---
    return jsonify({'answer': my_answer, 'chat_id': chat_id, 'qa_id': new_record_id}), 200

@app.route('/clear', methods=['POST'])
def clear():
    chat_id = session.get('chat_id')
    if 'chat_id' in session:
        session.pop('chat_id', None)
        if chat_id:
            print(f"Clearing memory for chat_id: {chat_id} on user request.")
            reset_user_memory(chat_id)
    return '', 200

def generate_id():
    return str(uuid.uuid4())

# --- 修改後的 /record_qa 路由 ---
@app.route('/record_qa', methods=['POST'])
def record_qa():
    db = None
    try:
        db = pymysql.connect(**DB_CONFIG)
        cursor = db.cursor()
        data = request.get_json()

        # --- 修改重點 3: 只接收紀錄ID和讚/倒讚狀態 ---
        qa_id = data.get('qa_id_')
        thumbs = data.get('thumbs_')

        if not all([qa_id, thumbs]):
            return jsonify({"error": "Missing qa_id or thumbs for update"}), 400

        # --- 修改重點 4: 使用ID來精確更新，不再比對文字 ---
        # 假設您的主鍵欄位是 'id'
        sql = "UPDATE qa_table SET thumbs = %s WHERE id = %s"
        
        rows_affected = cursor.execute(sql, (thumbs, qa_id))
        db.commit()

        if rows_affected > 0:
            print(f"Record ID {qa_id} thumbs updated to '{thumbs}'.")
            return jsonify({"message": "Thumbs updated successfully"}), 200
        else:
            print(f"Thumbs update failed for record ID {qa_id}. Record not found.")
            return jsonify({"error": "No matching record found to update"}), 404

    except pymysql.Error as e:
        if db: db.rollback()
        print(f"Database error in record_qa: {e}")
        return jsonify({"error": "Database error occurred", "details": str(e)}), 500
    except Exception as e:
        if db: db.rollback()
        print(f"An unexpected error occurred in record_qa: {e}")
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500
    finally:
        if db: db.close()

if __name__ == '__main__':
    app.run(debug=True)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80, debug=True )   開80port使用