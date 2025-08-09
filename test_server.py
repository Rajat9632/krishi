from flask import Flask, request

app = Flask(__name__)

# This route serves the simple webpage with a button
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head><title>POST Test</title></head>
    <body>
        <h2>Minimal Test Form</h2>
        <p>Clicking this button will send a POST request.</p>
        <form action="/test" method="post">
            <input type="submit" value="Send Test POST Request">
        </form>
    </body>
    </html>
    '''

# This route only accepts the POST request from the button
@app.route('/test', methods=['POST'])
def test_route():
    return "SUCCESS! The POST request was received."