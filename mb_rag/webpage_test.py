import gradio as gr
import os
from mb_rag.chatbot.basic import conversation_model, load_env
from mb_rag.rag.basic import RAG
import json

# Load environment variables
load_env('.env')

# Initialize RAG and chatbot
rag = RAG()
chatbot = None
current_conversation = None
conversations = {}

def load_conversations():
    if os.path.exists('conversations.json'):
        with open('conversations.json', 'r') as f:
            return json.load(f)
    return {}

def save_conversations():
    with open('conversations.json', 'w') as f:
        json.dump(conversations, f)

conversations = load_conversations()

def start_new_chat(context, question, api_key):
    global chatbot, current_conversation
    os.environ['OPENAI_API_KEY'] = api_key
    chatbot = conversation_model(context=context, question=question)
    current_conversation = {'context': context, 'messages': [{'role': 'human', 'content': question}, {'role': 'ai', 'content': chatbot.get_last_message()}]}
    conversations[context] = current_conversation
    save_conversations()
    return chatbot.get_last_message()

def chat(message, history):
    global chatbot, current_conversation
    if chatbot is None:
        return "Please start a new chat first."
    response = chatbot.add_message(message)
    current_conversation['messages'].append({'role': 'human', 'content': message})
    current_conversation['messages'].append({'role': 'ai', 'content': response})
    save_conversations()
    return response

def load_chat(context):
    global chatbot, current_conversation
    if context in conversations:
        current_conversation = conversations[context]
        chatbot = conversation_model(context=current_conversation['context'], question=current_conversation['messages'][0]['content'])
        for msg in current_conversation['messages'][2:]:
            if msg['role'] == 'human':
                chatbot.add_message(msg['content'])
        return gr.Chatbot(value=[(msg['content'], current_conversation['messages'][i+1]['content']) for i, msg in enumerate(current_conversation['messages'][::2])])
    return gr.Chatbot(value=[])

def rag_search(query):
    results = rag.search(query)
    return "\n\n".join([f"Title: {r['title']}\nContent: {r['content'][:200]}..." for r in results])

with gr.Blocks() as demo:
    gr.Markdown("# RAG and Chatbot Application")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Message")
            send = gr.Button("Send")
        
        with gr.Column(scale=1):
            context = gr.Textbox(label="Context/Title")
            question = gr.Textbox(label="Initial Question")
            api_key = gr.Textbox(label="OpenAI API Key", type="password")
            start_chat = gr.Button("Start New Chat")
            
            chat_list = gr.Dropdown(label="Previous Chats", choices=list(conversations.keys()))
            load_chat_btn = gr.Button("Load Selected Chat")
            
            rag_query = gr.Textbox(label="RAG Search Query")
            rag_search_btn = gr.Button("Search")
            rag_results = gr.Textbox(label="Search Results")

    send.click(chat, inputs=[msg, chatbot], outputs=[chatbot])
    start_chat.click(start_new_chat, inputs=[context, question, api_key], outputs=[chatbot])
    load_chat_btn.click(load_chat, inputs=[chat_list], outputs=[chatbot])
    rag_search_btn.click(rag_search, inputs=[rag_query], outputs=[rag_results])

if __name__ == "__main__":
    demo.launch()
