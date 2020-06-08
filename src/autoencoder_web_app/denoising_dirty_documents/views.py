from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import keras
from keras.models import load_model
import cv2
import tensorflow as tf
from tensorflow import Graph, Session
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array

# Create your views here.

RESIZE_WIDTH = 540
RESIZE_HEIGHT = 258

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=load_model('./models/denoising_dirty_documents/denoising_autoencoder.h5')


def index(request):
    context = {}
    template = loader.get_template('denoising_dirty_documents/denoising_dirty_documents_index.html')
    return HttpResponse(template.render(context, request))


def denoiseImage(request):
    # Uploading and saving an image in media files
    fileObj = request.FILES['filePath']

    fs = FileSystemStorage(location="media/denoising_dirty_documents")
    fileName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(fileName)

    # Preprocessing the uploaded image
    oImgPath = os.path.join("media/denoising_dirty_documents", fileName)
    oImg = load_img(oImgPath, grayscale=True, color_mode='grayscale', target_size=(RESIZE_HEIGHT, RESIZE_WIDTH))
    oImgNp = img_to_array(oImg)
    oReshapedImgNp = oImgNp.reshape(1, RESIZE_HEIGHT, RESIZE_WIDTH, 1)
    oNormalizedImgNp = oReshapedImgNp.astype('float32')/255.0

    # Predictions on the uploaded image
    with model_graph.as_default():
        with tf_session.as_default():
            pred=model.predict(oNormalizedImgNp)

    # Predictions are plotted
    pred_img= (pred*255.0).reshape(RESIZE_HEIGHT, RESIZE_WIDTH)
    plt.imshow(np.squeeze(pred_img), cmap='gray')
 
    # Saving the autoencoder output
    resultPathName =os.path.join("media/denoising_dirty_documents", "denoising_dirty_documents_result_"+fileName)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(resultPathName, bbox_inches='tight', pad_inches = 0)

    # Retrieving result image file path
    s = FileSystemStorage(location="media/denoising_dirty_documents")
    resultPathName = s.url("denoising_dirty_documents_result_"+fileName)

    x = filePathName.split('/')
    y = resultPathName.split('/')
    newfilePathName = "/"+x[1]+"/denoising_dirty_documents/"+x[2]
    newresultPathName = "/"+y[1]+"/denoising_dirty_documents/"+y[2]
    
    # Context building and template rendering
    context = {'filePathName' : newfilePathName, 
               'resultPathName' : newresultPathName}

    template = loader.get_template('denoising_dirty_documents/denoising_dirty_documents_result.html')
    return HttpResponse(template.render(context, request))


def viewDb(request):
    # Creating Db for gallery
    listOfImgs = os.listdir('./media/denoising_dirty_documents/')
    listofImgsPath = [i for i in listOfImgs]
    newlistofImgsPath = []
    for i in listOfImgs:
        p = i.split('_')
        print(p)
        if (len(p)>1) and ("result" in p):
            #Not adding to list of DBView images
            pass
        elif len(p)==1:
            #Adding to list of DBView images
            newlistofImgsPath.append('/media/denoising_dirty_documents/'+i)
        elif len(p)>1 and p[-2]!="result":
            #Adding to list of DBView images
            newlistofImgsPath.append('/media/denoising_dirty_documents/'+i)

    # Context building and template rendering
    context = {'listofImgsPath' : newlistofImgsPath}
    template = loader.get_template('denoising_dirty_documents/denoising_dirty_documents_viewDb.html')
    return HttpResponse(template.render(context, request))



