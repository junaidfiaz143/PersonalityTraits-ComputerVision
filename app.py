from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera

app = Flask(__name__)

global props
props = []

@app.route('/')
def index():
	return render_template('index.html')

def gen(camera):
	while True:
		frame, properties = camera.get_frame()
		print(properties)

		global props
		props = properties
		
		yield (b'--frame\r\n'b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update')
def update():
	global props
	return jsonify({"props": props})

if __name__ == '__main__':
	app.run(debug=True)