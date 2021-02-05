from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch
from flask import Flask, request, Response, jsonify
from flask import Flask, render_template, request, Response, send_file, jsonify

from torch.nn import functional as F
from queue import Queue, Empty
import time
import threading

# Server & Handling Setting
app = Flask(__name__)

requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1


tokenizer = AutoTokenizer.from_pretrained("jihopark/colloquial")
model = AutoModelWithLMHead.from_pretrained("jihopark/colloquial", return_dict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Queue 핸들링
def handle_requests_by_batch():
    while True:
        requests_batch = []
        while not (len(requests_batch) >= BATCH_SIZE):
            try:
                requests_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue

            for requests in requests_batch:
                # requests['output'] = run_word(requests['input'][0], requests['input'][1])
                # if len(requests['input']) == 2 :
                # elif len(requests['input']) == 3 :
                # requests['output'] = run_model(requests['input'][0][0], requests['input'][0][1], requests['input'][0][2])
                requests['output'] = run_model(requests['input'][0]+":")#, requests['input'][0][1], requests['input'][0][2])


# 쓰레드
threading.Thread(target=handle_requests_by_batch).start()

def run_model(prompt, num=1, length=30):
    try:
        prompt = prompt.strip()
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # input_ids also need to apply gpu device!
        input_ids = input_ids.to(device)

        min_length = len(input_ids.tolist()[0])
        length += min_length

        # model = models[model_name]
        sample_outputs = model.generate(input_ids, pad_token_id=50256,
                                        do_sample=True,
                                        max_length=300,

                                        top_k=50,
                                        num_return_sequences=num)
        generated_texts = ""
        for i, sample_output in enumerate(sample_outputs):
            output = tokenizer.decode(sample_output.tolist(),skip_special_tokens=False)
            generated_texts+= output+'\n'
        generated_texts.replace("<br/>",'\n')
        generated_texts.replace("<br />",'\n')
        return generated_texts

    except Exception as e:
        print(e)
        return 500



# @app.route("/gpt2-recipes-maker/", methods=['POST'])

@app.route("/api/", methods=['GET'])
def generate():

    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'error': 'Too Many Requests'}), 429

    try:
        args = []

        review=request.args.get('input')

        args.append(review)

    except Exception:
        print("Empty Text")
        return Response("fail", status=400)

    req = {
        'input': args
    }
    requests_queue.put(req)

    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)

    return req['output']


# Health Check

@app.route('/healthz')
def health():
    return "ok", 200

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
