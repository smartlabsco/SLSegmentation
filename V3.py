import os
import numpy as np
import logging
import logging.config
import json
import base64
import io
from faceModule.faceParser import FaceParser
from licenseModule.licenseParser import LicenseParser
from PIL import Image
from flask import Flask, request, jsonify, send_file

# log setting
log_config = json.load(open('log.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)

# GPU SETTING
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU 1 to use

# flask app
app = Flask(__name__)
return_status = {
    'notfound': 404,
    'invalid': 403,
    'ok': 200,
    'error': 500
}

# apiKey list
auth_file = "auth.ini"
with open(auth_file) as f:
    auth_list = f.read().splitlines()

# load parser
faceparser = FaceParser()
licenseparser = LicenseParser()

@app.route('/')
def welcome():
    return 'welcome smartlabs openAPI'

@app.route('/parsingLicense', methods=['POST'])
def parsingLicense():
    try:
        # check auth
        auth()

        # receive image file and read the image via file.stream
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        # parsing
        output_img = licenseparser.parsing(img, threshold=0.5, overlay=False, view=False) # numpy.Array
        output_str = base64String(output_img)

        return jsonify({'msg': 'success', 'img_size': [output_img.shape[0], output_img.shape[1]], 'img_data': output_str}), return_status['ok']
    except Exception as e:
        logger.error(e)
        return jsonify({'msg': e.__str__()}), return_status['error']

@app.route('/parsingFace', methods=['POST'])
def parsingFace():
    try:
        # check auth
        auth()

        # receive image file and read the image via file.stream
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')

        # parsing
        output_img = faceparser.parsing(img, overlay=False, view=False)
        output_str = base64String(output_img)

        return jsonify({'msg': 'success', 'img_size': [output_img.shape[0], output_img.shape[1]], 'img_data': output_str}), return_status['ok']
    except Exception as e:
        logger.error(e)
        return jsonify({'msg': e.__str__()}), return_status['error']

def auth():
    if 'apiKey' in request.headers:
        if request.headers['apiKey'] not in auth_list:
            raise Exception('auth error')
    else:
        raise Exception('auth error')

# numpyArray to PIL Image --> PIL Image to base64String
def base64String(output_img):
    # PIL Image to base64 string using io.BytesIO(memory area)
    oImage = Image.fromarray(output_img.astype('uint8'))
    in_mem_file = io.BytesIO()
    oImage.save(in_mem_file, format="PNG")
    base64_encoded_result_bytes = base64.b64encode(in_mem_file.getvalue())
    base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
    return base64_encoded_result_str


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=17776, debug=True)
