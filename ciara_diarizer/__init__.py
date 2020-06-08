from pathlib import Path

from flask import Flask

app = Flask(__name__)

from ciara_diarizer import routes
