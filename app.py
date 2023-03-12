from flask import Flask, Response, request, render_template
import os
import predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER']='./static'

def get_file(filename):
    try:
        return open(filename).read()
    except IOError as e:
        return str(e)
    

@app.route("/", methods=['GET', 'POST'])
def home():
    imagename= None
    predicted_name= None
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file in the form!"
        image = request.files['image']
        path = os.path.join(app.config['UPLOAD_FOLDER'],image.filename)
        imagename = image.filename 
        image.save(path)
        ext = image.filename.split('.')[-1]
        predicted_name = '.'.join([image.filename.split('.')[0]+"_predicted", ext])
        predicted_path = os.path.join(app.config['UPLOAD_FOLDER'],predicted_name)
        print(f"predicted_path: {predicted_path}")
        predict.predict(path, predicted_path)

    return render_template("home.html", imagename=imagename, predicted_name=predicted_name) 
    

if __name__=='__main__':
    app.run(debug=True, host="0.0.0.0")
