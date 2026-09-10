from pathlib import Path
import numpy as np, tensorflow as tf
from PIL import Image
from flask import Flask,request,jsonify,send_from_directory
ROOT=Path(__file__).resolve().parent; MODEL=ROOT/'model'/'best.keras'; WEB=ROOT/'web'; app=Flask(__name__,static_folder=str(WEB),static_url_path='')
model=tf.keras.models.load_model(MODEL) if MODEL.exists() else None
CLASSES=['Grade A','Grade B','Grade C']
@app.get('/')
def home(): return send_from_directory(WEB,'Pyaz-Nyay-ML.html')
@app.post('/predict')
def predict():
    if model is None:return jsonify(error='Model not trained. Run python train.py first.'),503
    if 'image' not in request.files:return jsonify(error='No image supplied.'),400
    im=Image.open(request.files['image']).convert('RGB').resize((224,224)); x=np.asarray(im,dtype=np.float32)[None,...]; probs=model.predict(x,verbose=0)[0]; k=int(np.argmax(probs)); conf=float(probs[k]); grade=CLASSES[k] if conf>=.60 else 'Needs manual inspection'
    return jsonify(grade=grade,confidence=round(conf*100,2),probabilities={CLASSES[i]:round(float(probs[i])*100,2) for i in range(3)})
if __name__=='__main__': app.run(host='127.0.0.1',port=5000,debug=False)
