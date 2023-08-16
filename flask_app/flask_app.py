from flask import Flask,render_template,request
import os
import numpy as np
from numpy import array

import pickle
from matplotlib.image import imread
import skimage

app=Flask(__name__)

BASEPATH=os.getcwd()
UPLOAD_PATH=os.path.join(BASEPATH,"static/uploads/")
# loading models
model_path=os.path.join('../','model.pickle')

model=pickle.load(open(model_path,'rb'))


@app.route('/',methods=['GET','POST'])

def index():
    if(request.method=="POST"):
        upload_file=request.files['my_image']
        filename=upload_file.filename
        print("Uploaded File is ",filename)
        #extension of file
        #allow only .jpg,.jpeg,.png
        ext=filename.split('.')[-1]
        print("The extension of filename is ",ext)
      
        if(ext.lower() in ['png','jpeg','jpg']):
            
            #send to ML model
            path_save=os.path.join(UPLOAD_PATH,filename)
            upload_file.save(path_save)
            print(upload_file)
            sample_img=imread(upload_file)
            # a = array(sample_img)
            # print(a.shape)
            sample_img=rgb2gray(sample_img)
            # a = array(sample_img)
            # print(a.shape)
            sample_img=skimage.transform.resize(sample_img,(32,32))
            sample_img=sample_img.reshape(1,32,32,1)
            # a = array(sample_img)
            # print(a.shape)
            results=model.predict(sample_img)
            print(results)
            results[0]=np.round(results[0],2)
            top_dict = dict()
            top_dict[0]=results[0][0]
            top_dict[1]=results[0][1]
            top_dict[2]=results[0][2]
            top_dict[3]=results[0][3]
            top_dict[4]=results[0][4]
            top_dict[5]=results[0][5]
            top_dict[6]=results[0][6]
            top_dict[7]=results[0][7]
            top_dict[8]=results[0][8]
            top_dict[9]=results[0][9]
            
            return render_template('upload.html',fileupload=True,data=top_dict,image_filename=filename)
        else:
            print(ext," file extension not allowed")

    return render_template('upload.html')
def rgb2gray(rgb):
    try:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray
    except:
        return rgb
# def get_height(path):
#     img=skimage.io.imread(path)
#     h,w,_=img.shape
#     aspect=h/w
#     width=200
#     height=aspect*width
#     return height
if __name__=='__main__':
    app.run(debug=True)