from flask import Flask, request, jsonify
from transformers import pipeline
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

try:
    generator = pipeline('text-generation', model='gpt2')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error occurred while loading the model: {e}")


@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data['prompt']
        logging.debug(f"Received prompt: {prompt}")
        
        generated_text = generator(prompt, max_length=50, num_return_sequences=1)['generated_text']
        logging.debug(f"Generated text: {generated_text}")
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        return jsonify(f'error: {str(e)}'), 500


if __name__ == '__main__':
    app.run(debug=True)
