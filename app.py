from flask import Flask

app = Flask(__name__)

@app.route('/')
def homepage():
    return "Hello from Vercel!"

# A catch-all route to prevent crashes from favicon requests, etc.
@app.route('/<path:path>')
def catch_all(path):
    return "Hello from Vercel!"