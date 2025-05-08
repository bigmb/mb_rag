from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import base64
from langchain.schema import HumanMessage

'''
docker build -f dockerfile_google -t gemini-api .
docker run -p 5001:5000 -e GOOGLE_API_KEY='api_key' gemini-api

docker build -f dockerfile_openai -t openai-api .
docker run -p 5001:5000 -e OPENAI_API_KEY='api_key' openai-api  

## internal port is 5000
## external port is 5001
'''

app = Flask(__name__)

if os.getenv("MODEL") == "google":
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "default-none")
    chat = ChatGoogleGenerativeAI(model=GOOGLE_MODEL_NAME)
elif os.getenv("MODEL") == "openai":
    from langchain_openai import ChatOpenAI
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default-none")
    chat = ChatOpenAI(model=OPENAI_MODEL_NAME)
else:
    raise ValueError("Invalid model specified")

### curl -X POST http://localhost:5001/api/text   -H "Content-Type: application/json"   -d '{"prompt": "What is the capital of India?"}'
@app.route("/api/text", methods=["POST"])
def handle_text():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = chat.invoke(prompt)
    return jsonify({"response": response.content if hasattr(response, 'content') else str(response)})

### curl -X POST http://localhost:5001/api/image   -F "prompt=Describe the contents of the image"   -F "image=@/home/winnow/Downloads/mt_image1.jpg"
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
