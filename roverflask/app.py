from flask import Flask, render_template, request, redirect, url_for, make_response
import time
app = Flask(__name__) #set up flask server
#when the root IP is selected, return index.html page
@app.route('/')
def index():
    return render_template('/index.html')
#recieve which pin to change from the button press on index.html
#each button returns a number that triggers a command in this function
#
#Uses methods from motors.py to send commands to the GPIO to operate the motors
@app.route('/<changepin>', methods=['POST'])
def reroute(changepin):
    if changepin == '1':
        name = 'Moving forward'
    elif changepin == '2':
        name = 'Turning left'
    elif changepin == '3':
        name = 'Stopping'
    elif changepin == '4':
        name = 'Turning right'
    elif changepin == '5':
        name = 'Moving backward'
    return name

app.run(debug=True, host='0.0.0.0', port=8000) #set up the server in debug mode to the port 8000