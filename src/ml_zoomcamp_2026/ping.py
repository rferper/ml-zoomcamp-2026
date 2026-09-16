from flask import Flask

app = Flask('ping') # creates the application and gives it a name

@app.route('/ping', methods=['GET']) # it is a decorator: it tells Flask that the function below should ahndle GET requests to the /ping address. This address is called route.

def ping():
    return "PONG" # returns the response body - the string PONG

if __name__ == "__main__": # block runs the development server, but only when we execute the file directly (python ping.py), not when we import it from another file
    app.run(debug=True, host='0.0.0.0', port=9696) # host='0.0.0.0' instead means "accept requests coming from the network", which is what we want once the service runs on a server. For local development either works.
