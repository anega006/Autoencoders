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

RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224
THRESHOLD_BINARY = 0.5

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=load_model('./models/semantic_hair_segmentation/semantic_segmentation_autoencoder.hdf5')


def index(request):
    context = {}
    template = loader.get_template('semantic_hair_segmentation/semantic_hair_segmentation_index.html')
    return HttpResponse(template.render(context, request))


def segmentImage(request):
    # Uploading and saving an image in media files
    fileObj = request.FILES['filePath']

    fs = FileSystemStorage(location="media/semantic_hair_segmentation")
    fileName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(fileName)

    # Preprocessing the uploaded image
    oImgPath = os.path.join("media/semantic_hair_segmentation", fileName)
    oImg = load_img(oImgPath, grayscale=False, color_mode='rgb', target_size=(RESIZE_HEIGHT, RESIZE_WIDTH))
    oImgNp = img_to_array(oImg)
    oReshapedImgNp = oImgNp.reshape(1, RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    oNormalizedImgNp = oReshapedImgNp.astype('float32')/255.0

    # Predictions on the uploaded image
    with model_graph.as_default():
        with tf_session.as_default():
            pred=model.predict(oNormalizedImgNp)
    pred= pred.reshape(RESIZE_HEIGHT, RESIZE_WIDTH, 1)
    oNormalizedImgNp = oNormalizedImgNp.reshape(RESIZE_HEIGHT, RESIZE_WIDTH, 3)

    # Creating Binary segmentation map and overlay
    preds_hair_segment_binary = (pred > THRESHOLD_BINARY).astype(np.uint8)
    mask = np.zeros((RESIZE_HEIGHT, RESIZE_WIDTH,3))
    mask[:, :, 2] = preds_hair_segment_binary[:, :, 0]
    added_image = cv2.addWeighted(oNormalizedImgNp, 0.8, mask, 0.2, 0, dtype = cv2.CV_32F)
    
    # Predictions are plotted
    fig, ax = plt.subplots(1,2, figsize=(10,9))
    ax[0].imshow(np.squeeze(preds_hair_segment_binary), cmap='gray')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_frame_on(False)
    ax[1].imshow(np.squeeze(added_image))
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_frame_on(False)

    fig.tight_layout(pad=1.0) 
 
    # Saving the autoencoder output
    resultPathName =os.path.join("media/semantic_hair_segmentation", "semantic_hair_segmentation_result_"+fileName)
    plt.savefig(resultPathName, bbox_inches='tight', pad_inches = 0)

    # Retrieving result image file path
    s = FileSystemStorage(location="media/semantic_hair_segmentation")
    resultPathName = s.url("semantic_hair_segmentation_result_"+fileName)

    x = filePathName.split('/')
    y = resultPathName.split('/')
    newfilePathName = "/"+x[1]+"/semantic_hair_segmentation/"+x[2]
    newresultPathName = "/"+y[1]+"/semantic_hair_segmentation/"+y[2]
    
    # Context building and template rendering
    context = {'filePathName' : newfilePathName, 
               'resultPathName' : newresultPathName}

    template = loader.get_template('semantic_hair_segmentation/semantic_hair_segmentation_result.html')
    return HttpResponse(template.render(context, request))


def viewDb(request):
    #Creating db for gallery
    listOfImgs = os.listdir('./media/semantic_hair_segmentation/')
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
            newlistofImgsPath.append('/media/semantic_hair_segmentation/'+i)
        elif len(p)>1 and p[-2]!="result":
            #Adding to list of DBView images
            newlistofImgsPath.append('/media/semantic_hair_segmentation/'+i)

    # Context building and template rendering
    context = {'listofImgsPath' : newlistofImgsPath}
    template = loader.get_template('semantic_hair_segmentation/semantic_hair_segmentation_viewDb.html')
    return HttpResponse(template.render(context, request))
