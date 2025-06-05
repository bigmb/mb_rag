from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import base64
from PIL import Image
import io
from langchain.schema import HumanMessage

app = Flask(__name__)

# Load the appropriate model based on environment variable
if os.getenv("MODEL") == "qwen":
    QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3.1-7B")
    model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
elif os.getenv("MODEL") == "kimi":
    KIMI_MODEL_NAME = os.getenv("KIMI_MODEL_NAME", "HuggingFaceH4/kimi-moonshot")
    model = AutoModelForCausalLM.from_pretrained(KIMI_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(KIMI_MODEL_NAME)
else:
    raise ValueError("Invalid model specified")

@app.route("/api/text", methods=["POST"])
def handle_text():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"response": response})

@app.route("/api/image", methods=["POST"])
def handle_image():
    prompt = request.form.get("prompt")
    image_file = request.files.get("image")  

    if not prompt or not image_file:
        return jsonify({"error": "Prompt and image are required"}), 400

    try:
        # For Hugging Face models that support image inputs
        # This implementation may need to be adjusted based on the specific model capabilities
        image = Image.open(io.BytesIO(image_file.read()))
        
        # Process with model - implementation depends on model's multimodal capabilities
        # This is a placeholder - actual implementation will depend on the model's API
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
