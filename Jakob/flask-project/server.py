from flask import Flask, render_template, request
from weather import get_current_weather
from waitress import serve
import wikipedia

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/mushroom')
def get_mushroom():
    mushroom_name = request.args.get('mushroom')

    # Check for empty strings or string with only spaces
    if not bool(mushroom_name.strip()):
        return render_template('mushroom-not-found.html')
    
    mushroom_data = wikipedia.summary(mushroom_name)
    print(mushroom_data)
    
    if len(mushroom_data) < 1:
        return render_template('mushroom-not-found.html')

    return render_template(
        "mushroom.html",
        mushroomData=mushroom_data
    )


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
