"""
智能客服 Agent 主应用
Flask Web 服务入口
LangChain 版本
"""

import uuid
import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from modules.ai_client import AIClient
from modules.store.vector_store import VectorStore
from modules.assistant import Agent
from modules.tools import get_all_tools
from modules.prompt import create_chat_prompt

if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    import codecs
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

os.environ['PYTHONIOENCODING'] = 'utf-8'

load_dotenv()

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
CORS(app)

assistant_instance = None
vector_store_instance = None
sessions = {}


def init_system():
    """初始化系统组件"""
    global assistant_instance, vector_store_instance

    print("=" * 50)
    print("智能客服系统启动中... (LangChain 版本)")
    print("=" * 50)

    print("\n[1/3] 初始化 AI 客户端...")
    try:
        ai_client = AIClient(config_path="config.json")
        print("AI 客户端初始化完成")
    except Exception as e:
        print("AI 客户端初始化失败: {}".format(e))
        raise

    print("\n[2/3] 初始化知识库...")
    try:
        vector_store_instance = VectorStore(ai_client=ai_client)
        kb_data = vector_store_instance.init_knowledge_base()
        if kb_data["status"] == "error":
            print(f"知识库加载失败: {kb_data.get('message', '未知错误')}")
        elif kb_data["status"] == "empty":
            print("知识库为空，将跳过 RAG 检索")
        else:
            print(f"知识库初始化完成，共 {kb_data['count']} 个向量")
    except Exception as e:
        print("知识库初始化警告: {}".format(e))

    print("\n[3/3] 初始化 AI 助手...")
    try:
        tools = get_all_tools()

        assistant_instance = Agent(options={
            "prompt": create_chat_prompt(),
            "vectorStore": vector_store_instance,
            "tools": tools,
            "aiClient": ai_client
        })
        print("AI 助手初始化完成")
    except Exception as e:
        print("AI 助手初始化失败: {}".format(e))
        raise

    print("\n" + "=" * 50)
    print("智能客服系统就绪!")
    print("=" * 50)
    print("\n服务地址: http://localhost:5000")
    print("API 文档:")
    print("  GET  /start  - 检查服务状态")
    print("  POST /chat   - 发送对话请求")
    print("=" * 50 + "\n")


@app.route('/start', methods=['GET'])
def start():
    """检查服务状态"""
    try:
        status = {
            "status": "ready",
            "message": "客服系统已就绪 (LangChain)",
            "model": "deepseek-v4-pro",
            "knowledge_base": vector_store_instance is not None
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """处理对话请求"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "error": "缺少 message 字段"
            }), 400

        user_message = data['message']
        session_id = data.get('session_id', str(uuid.uuid4()))

        print("\n[对话请求] Session: {}".format(session_id))
        print("用户: {}".format(user_message))

        result = assistant_instance.process_message(session_id, user_message)

        response = {
            "reply": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
            "session_id": session_id,
            "finished": False
        }

        return jsonify(response)

    except Exception as e:
        print("对话处理异常: {}".format(e))
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == '__main__':
    init_system()
    app.run(host='0.0.0.0', port=5000, debug=True)
