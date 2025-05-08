from transformers import AutoModelForCausalLM, AutoTokenizer
import os

QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3.1-7B")

model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

@app.route("/api/text", methods=["POST"])
def handle_text():
    data = request.json
    prompt = data.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    response = model.generate(**tokenizer(prompt, return_tensors="pt").input_ids)
    return jsonify({"response": tokenizer.decode(response[0], skip_special_tokens=True)})

# @app.route("/api/image", methods=["POST"])
# def handle_image():
#     prompt = request.form.get("prompt")
#     image_file = request.files.get("image")  

#     if not prompt or not image_file:
#         return jsonify({"error": "Prompt and image are required"}), 400

#     try:
#         image_bytes = image_file.read()
#         image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
#         image_prompt_create = [{
#             "type": "image_url",
#             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
#         }]
#         prompt_new = [{"type": "text", "text": prompt}, *image_prompt_create]
#         message = HumanMessage(content=prompt_new)
#         response = chat.invoke([message])

#         return jsonify({
#             "response": response.content if hasattr(response, 'content') else str(response)
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    