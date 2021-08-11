from flask import Flask,render_template,request
import os
import base64
import requests
import torch
import numpy as np
import argparse
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.transformer import Transformer
from skimage import io



BASE_PATH=os.getcwd()
UPLOAD_PATH=os.path.join(BASE_PATH,'static/Upload/')
RESULT_PATH=os.path.join(BASE_PATH,'static\cartoonized_img')
MODEL_PATH=os.path.join(BASE_PATH,'static/pretrained_models/')


styles = ["Hosoda", "Hayao", "Shinkai", "Paprika"]

models = {}

for style in styles:
    model = Transformer()
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, style + '_net_G_float.pth')))
    model.eval()
    models[style] = model

styles='Hosoda'
LOAD_SIZE=400

app=Flask(__name__)

@app.errorhandler(404)
def error404(error):
    message='ERROR 404 OCCURED.Page not found Please go to the home age and try again'
    return render_template('error.html',message=message)

@app.errorhandler(405)
def error405(error):
    message='ERROR 405 OCCURED.Method not found '
    return render_template('error.html',message=message)

@app.errorhandler(505)
def error505(error):
    message='INTERNAL ERROR OCCURED. Please go to the home age and try again'
    return render_template('error.html',message=message)

# print(app)
@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        upload_files=request.files['image_name']
        fname=upload_files.filename
        print('The file that has been uploaded is',fname)
        ext=fname.split('.')[-1]
        print('The extension of the file uploaded is ',ext)
        if ext.lower() in ['jpg','jpeg','png']:
            path_saved=os.path.join(UPLOAD_PATH,fname)
            upload_files.save(path_saved)
        
            results=transform(models,style,path_saved)
           
            out_dict={'output':results}
            image=str(out_dict['output'])
            image=image[image.find(',')+1:] 
            image=base64.b64decode(image +"===")
            # image=Image.open(BytesIO(dec))
            output_fname='some_img.png'
            result_saved=os.path.join(RESULT_PATH,output_fname)
            with open(result_saved, 'wb') as image_data:
                  image_data.write(image)
            hei=getheight(result_saved)
            #image.close()
            #image.show()
            #print(type(image))

            #results.save(result_saved)
            return render_template('upload.html',extension=False,fileupload=True,image_filename=output_fname,height=hei)
        else:
            print('Only upload images with the extension .jpg,.jpeg or .png')
            return render_template('upload.html',extension=True,fileupload=False)
        
    else:
        return render_template('upload.html',extension=False,fileupload=False)
        

    
def transform(models, style, input, load_size=LOAD_SIZE, gpu=-1):
    model = models[style]

    if gpu > -1:
        model.cuda()
    else:
        model.float()

    input_image = Image.open(input).convert("RGB")
    h, w = input_image.size

    ratio = h * 1.0 / w

    if ratio > 1:
        h = load_size
        w = int(h * 1.0 / ratio)
    else:
        w = load_size
        h = int(w * ratio)

    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    input_image = -1 + 2 * input_image
    if gpu > -1:
        input_image = Variable(input_image).cuda()
    else:
        input_image = Variable(input_image).float()

    
    with torch.no_grad():
        output_image = model(input_image)[0]
    

    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    output_image = output_image.numpy()

    output_image = np.uint8(output_image.transpose(1, 2, 0) * 255)
    output_image = Image.fromarray(output_image)
    output_image=img_to_base64_str(output_image)

    return output_image


def getheight(path):
    img=io.imread(path)
    h,w,_=img.shape
    aspect_ratio=h/w
    given_width=300
    height=aspect_ratio*given_width
    return height


def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str