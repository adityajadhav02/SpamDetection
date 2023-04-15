from flask import Flask,flash, render_template, request,redirect,send_file
import pandas as pd
import numpy as np
import pickle
classifier = pickle.load(open('./spamClassificationModel.pkl', 'rb'))
cv = pickle.load(open('countVectModel.pkl','rb'))
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    output=""
    # for text    
    if request.form['message']:
          message = request.form['message']
          data = [message]
          if data:
            vect = cv.transform(data).toarray()
            prediction = classifier.predict(vect)
            output=prediction[0]
            if output=='spam':
                output="Spam Text Detected" 
            elif output=='ham':
                output= "Not a Spam Text"

    # for file
    elif request.files['file']:
        f=request.files['file']
        
        if(f.filename.endswith('.csv')):
            data=pd.read_csv(f,encoding = "ISO-8859-1")
            data=data[data.columns[0]]
            if not data.empty:
                vect = cv.transform(data).toarray()
                prediction = classifier.predict(vect)
                
                df = pd.DataFrame()
                df['text']=np.array(data)
                df['prediction']=prediction
                print(df)
                df.to_csv('prediction.csv')
                return send_file('prediction.csv', as_attachment=True) 
        elif(f.filename.endswith('.zip')):
            import zipfile
            my_zip = zipfile.ZipFile(f) 
            text=[]
            for file in my_zip.namelist():
                if my_zip.getinfo(file).filename.endswith('.txt'):
                    text.append( my_zip.read(file).decode('utf-8'))
            df = pd.DataFrame()
            df['text']=text
            data=df[df.columns[0]]
            if not data.empty:
                vect = cv.transform(data).toarray()
                prediction = classifier.predict(vect)
                
                df['prediction']=prediction
                
                df.to_csv('prediction.csv')
                return send_file('prediction.csv', as_attachment=True) 
        else:    
            output="Enter File in given Format Only"
    else:
         output="Enter in specified input options only"
         return render_template('index.html', prediction_text='{}'.format(output))

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == '__main__':
	app.run(debug=True)