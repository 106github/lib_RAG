from flask import Flask, jsonify, request, session, render_template
# from lib_rag import test as test_lib, create_memory, reset_user_memory
from book_rag import test as test_lib, create_memory, reset_user_memory
# from mcut_rag import test as test_lib, create_memory, reset_user_memory
# from care_rag import test as test_lib, create_memory, reset_user_memory

from langchain.memory import ConversationBufferMemory 
import datetime
import uuid
import pymysql
import os 


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'hahaha')

@app.route('/')
def index():
    if 'chat_id' not in session:
        session['chat_id'] = generate_id()
        create_memory(session['chat_id'])
    return render_template('index.html', chat_id=session['chat_id'])

@app.route('/lib_test', methods=['POST'])
def lib_test():
    if not request.json or not 'question' in request.json:
        return jsonify({'error': '請輸入問題'}), 400

    my_question = request.json['question']

    chat_id = session.get('chat_id')


    if not chat_id:
        print("Warning: chat_id missing from session in /lib_test. Generating new one.")
        session['chat_id'] = generate_id()
        chat_id = session['chat_id']
        create_memory(chat_id) 



    my_answer = test_lib(chat_id, my_question)

   
    return jsonify({'answer': my_answer, 'chat_id': chat_id}), 200 

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

@app.route('/record_qa', methods=['POST']) 
def record_qa():
    db_host = "127.0.0.1"
    db_user = "root"
    db_passwd = "1234"
    db_name = "lib_qa"

    db = None 
    try:
        db = pymysql.connect(
            host=db_host,
            user=db_user,
            passwd=db_passwd,
            database=db_name,
            charset='utf8'
        )
        cursor = db.cursor()
        data = request.get_json()


        question = data.get('question_')
        answer = data.get('answer_')
        chat_id = data.get('chat_id_') 
        thumbs = data.get('thumbs_')
        current_time = datetime.date.today()

        if not question or not answer or chat_id is None or thumbs is None:
             print(f"Validation failed for record_qa. Received data: question='{question}', answer='{answer}', chat_id='{chat_id}', thumbs='{thumbs}'") # Added more detailed logging
             return jsonify({"error": "Missing or invalid data in request"}), 400

        cursor.execute("SELECT COUNT(*) FROM qa_table WHERE session_id = %s AND question = %s AND answer = %s LIMIT 1", (chat_id, question, answer))
        edit_thumb_count = cursor.fetchone()[0]

        if edit_thumb_count > 0:
            cursor.execute("UPDATE qa_table SET thumbs = %s WHERE session_id = %s AND question = %s AND answer = %s", (thumbs, chat_id, question, answer))
            db.commit()
            print(f"Thumbs updated successfully for chat_id: {chat_id}") 
            return jsonify({"message": "Thumbs updated successfully"}), 200
        else:
            cursor.execute("INSERT INTO qa_table(session_id, question, answer, thumbs, day_time) VALUES(%s, %s, %s, %s, %s)", (chat_id, question, answer, thumbs, current_time))
            db.commit()
            print(f"Record inserted successfully for chat_id: {chat_id}") 
            return jsonify({"message": "Record inserted successfully"}), 200

    except pymysql.Error as e: 
        if db:
            db.rollback() 
        print(f"Database error in record_qa for chat_id {chat_id}: {e}")
        return jsonify({"error": "Database error occurred", "details": str(e)}), 500 
    except Exception as e:
        if db:
             db.rollback() 
        print(f"An unexpected error occurred in record_qa for chat_id {chat_id}: {e}")
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500 
    finally:
        if db:
            db.close()

if __name__ == '__main__':
    app.run(debug=True)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80, debug=True )   開80port使用