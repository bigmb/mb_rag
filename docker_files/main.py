from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

app = Flask(__name__)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
MODEL_KEY = os.getenv("GOOGLE_API_KEY", "default-none")
chat = ChatGoogleGenerativeAI(model=MODEL_NAME)

### curl -X POST http://localhost:5000/api/text   -H "Content-Type: application/json"   -d '{"prompt": "What is the capital of India?"}'
@app.route("/api/text", methods=["POST"])
def handle_text():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = chat.invoke(prompt)
    return jsonify({"response": response.content if hasattr(response, 'content') else str(response)})

### curl -X POST http://localhost:5000/api/image   -F "prompt=Describe the contents of the image"   -F "image=@/home/winnow/Downloads/mt_image1.jpg"
@app.route("/api/image", methods=["POST"])
def handle_image():
    prompt = request.form.get("prompt")
    image_file = request.files.get("image")  

    if not prompt or not image_file:
        return jsonify({"error": "Prompt and image are required"}), 400

    try:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        image_prompt_create = [{
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        }]
        prompt_new = [{"type": "text", "text": prompt}, *image_prompt_create]
        message = HumanMessage(content=prompt_new)
        response = chat.invoke([message])

        return jsonify({
            "response": response.content if hasattr(response, 'content') else str(response)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
