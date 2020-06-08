from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from django.template import loader

from django.core.files.storage import FileSystemStorage, Storage
from django.conf import settings

import keras
from keras.models import load_model
import cv2
import tensorflow as tf
from tensorflow import Graph, Session
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array


RESIZE_WIDTH = 256
RESIZE_HEIGHT = 256
ENCODING_SIZE = 8*8*512
NEAREST_NEIGHBOURS_COUNT = 5

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=load_model('./models/reverse_image_search/autoencoder.hdf5')
        encoder=load_model('./models/reverse_image_search/encoder.hdf5')

pickle_in = open("./reverse_image_search/TestImgArrayNpDB/oTestImgArrayNp.pickle","rb")
oTestImgArrayNp = pickle.load(pickle_in) 


def index(request):
    context = {}
    template = loader.get_template('reverse_image_search/reverse_image_search_index.html')
    return HttpResponse(template.render(context, request))

def similarImage(request):
    # Uploading and saving an image in media files
    fileObj = request.FILES['filePath']

    fs = FileSystemStorage(location="media/reverse_image_search")
    fileName = fs.save(fileObj.name, fileObj)		#queryimage
    filePathName = fs.url(fileName)

    # Preprocessing the uploaded image
    oImgPath = os.path.join("media/reverse_image_search", fileName)
    oImg = load_img(oImgPath, grayscale=False, color_mode='rgb', target_size=(RESIZE_HEIGHT, RESIZE_WIDTH))
    oImgNp = img_to_array(oImg)
    oReshapedImgNp = oImgNp.reshape(1, RESIZE_HEIGHT, RESIZE_WIDTH, 3)
    oNormalizedImgNp = oReshapedImgNp.astype('float32')/255.0

    # Predictions on the uploaded image
    with model_graph.as_default():
        with tf_session.as_default():
            query_encoding = encoder.predict(oNormalizedImgNp)
            data_encoding = encoder.predict(oTestImgArrayNp)

    # Encodings of test images in database and query image
    oTestImgCodeNp = data_encoding.reshape(-1, ENCODING_SIZE)
    oQueryCodeNp = query_encoding.reshape(-1, ENCODING_SIZE)

    # Selecting Nearest N Neighbours
    nbrs = NearestNeighbors(n_neighbors=NEAREST_NEIGHBOURS_COUNT, metric='euclidean')
    nbrs.fit(oTestImgCodeNp)

    # Distances and indices of nearest neigbours from query image
    distances, indices = nbrs.kneighbors(oQueryCodeNp)
    print("Distances :", distances)
    print("Indices :", indices)
 
    oClosestImgs = oTestImgArrayNp[indices]
    oClosestImgs = oClosestImgs.reshape(-1,256,256,3)

    idx = indices[0]
    dist = distances[0]

    # Plotting the output
    fig, ax = plt.subplots(1,5, figsize=(10,9))

    for i in range(NEAREST_NEIGHBOURS_COUNT):
      ax[i].imshow(oTestImgArrayNp[idx[i]])
      ax[i].title.set_text("Score:"+"{0:.2f}".format(dist[i]))
      ax[i].get_xaxis().set_visible(False)
      ax[i].get_yaxis().set_visible(False)
      ax[i].set_frame_on(False)

    fig.tight_layout(pad=1.0) 
 
    # Saving the autoencoder output   
    resultPathName =os.path.join("media/reverse_image_search", "reverse_image_search_result_"+fileName)
    plt.savefig(resultPathName, bbox_inches='tight', pad_inches = 0)

    # Retrieving result image file path
    s = FileSystemStorage(location="media/reverse_image_search")
    resultPathName = s.url("reverse_image_search_result_"+fileName)

    x = filePathName.split('/')
    y = resultPathName.split('/')
    newfilePathName = "/"+x[1]+"/reverse_image_search/"+x[2]
    newresultPathName = "/"+y[1]+"/reverse_image_search/"+y[2]

    # Context building and template rendering
    context = {'filePathName' : newfilePathName, 
               'resultPathName' : newresultPathName}
    template = loader.get_template('reverse_image_search/reverse_image_search_result.html')
    return HttpResponse(template.render(context, request))


def viewDb(request):
    # Creating Db for gallery
    listOfImgs = os.listdir('./media/reverse_image_search/')
    listofImgsPath = [i for i in listOfImgs]
    newlistofImgsPath = []
    for i in listOfImgs:
        p = i.split('_')
        if (len(p)>1) and ("result" in p):
            #Not adding to list of DBView images
            pass
        elif len(p)==1:
            #Adding to list of DBView images
            newlistofImgsPath.append('/media/reverse_image_search/'+i)
        elif len(p)>1 and p[-2]!="result":
            #Adding to list of DBView images
            newlistofImgsPath.append('/media/reverse_image_search/'+i)

    # Context building and template rendering
    context = {'listofImgsPath' : newlistofImgsPath}
    template = loader.get_template('reverse_image_search/reverse_image_search_viewDb.html')
    return HttpResponse(template.render(context, request))






















