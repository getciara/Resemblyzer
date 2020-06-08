import json

from flask import request, Response

from ciara_diarizer import app, utils


@app.route('/')
def index():
    return Response(response="Hello Ciara Team! Diarizer is up & running", status=200)


@app.route('/health_check')
def health_check():
    return Response(response="success", status=200)


@app.route('/embed', methods=['POST'])
def match():
    data = json.loads(request.data)
    print(data)
    speaker_embed = utils.fingerprint_from_file(filepath=data.get('filepath'), segment=data.get('segment'))
    return Response(response="Speaker embedding created!", status=201)


@app.route('/diarize', methods=['GET'])
def summarize():
    # data = json.loads(request.data)
    response = None
    return Response(status=200, response=json.dumps(response))
