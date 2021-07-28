from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle
from features import *
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
	return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def index():
	if request.method == 'POST':
		try:
			web_link=str(request.form['link'])
			print(web_link)
			model = int(request.form['model'])
			print(model)

			data=featureExtraction(web_link)
			print(data)

			if model==2:
				filename = 'DecisionTree.pickle.dat'
				loaded_model = pickle.load(open(filename, 'rb'))
				prediction=loaded_model.predict([data])
				print('prediction is', prediction[0])
			elif model==1:
				filename = 'RandomForest.pickle.dat'
				loaded_model = pickle.load(open(filename, 'rb'))
				prediction=loaded_model.predict([data])
				print('prediction is', prediction[0])
			elif model==3:
				filename = 'Support_Vector.pickle.dat'
				loaded_model = pickle.load(open(filename, 'rb'))
				prediction=loaded_model.predict([data])
				print('prediction is', prediction[0])
			else:
				filename = 'XGBoostClassifier.pickle.dat'
				loaded_model = pickle.load(open(filename, 'rb'))
				prediction=loaded_model.predict(np.array([data]))
				print('prediction is', prediction[0])

			if prediction[0]==1:
				return render_template('result.html', pred =True  ,link=web_link )
			else:
				 return render_template('result.html', pred =False  ,link=web_link )
		except Exception as e:
			print('The Exception message is: ',e)
			return 'something is wrong'
	else:
		return render_template('index.html')
if __name__ == "__main__":
	app.run(debug=True)