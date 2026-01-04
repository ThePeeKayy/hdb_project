from flask import Flask, request
import lambda_functions as lambda_app

app = Flask(__name__)

def create_event(request, path):
    return {
        'httpMethod': request.method,
        'path': path,
        'headers': dict(request.headers),
        'body': request.get_data(as_text=True) if request.data else ''
    }

@app.route('/api/listings', methods=['GET', 'POST', 'OPTIONS'])
def listings():
    event = create_event(request, '/api/listings')
    response = lambda_app.lambda_handler(event, None)
    return (response['body'], response['statusCode'], response['headers'])

@app.route('/api/helper', methods=['POST', 'OPTIONS'])
def helper():
    event = create_event(request, '/api/helper')
    response = lambda_app.lambda_handler(event, None)
    return (response['body'], response['statusCode'], response['headers'])

@app.route('/', methods=['GET'])
def index():
    return 'Flask app is running!', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
