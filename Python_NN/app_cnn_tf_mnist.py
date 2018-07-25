import sys
import os
import numpy as np
from PIL import Image
from scipy import ndimage
import scipy

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable Deprecation Warnings
import warnings
warnings.filterwarnings("ignore")
 
###########################################################################
### Getting base64 image from NodeJS and convert it to numpy array with shape (1,28,28,1)
###########################################################################
raw_image_data_string = sys.argv[1]    # base64 image FROM NODE
model_path = sys.argv[2]               # path of the cached model FROM NODE

# clear base64 image data
string_to_remove = "data:image/jpeg;base64,"
image_base_64_string = raw_image_data_string.replace(string_to_remove, "")

import base64
import io
import matplotlib.image as mpimg

img = base64.b64decode(image_base_64_string)
img = io.BytesIO(img)
numpy_image = mpimg.imread(img, format='JPG')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def rescale_image(image):
    ## rescale
    i_width = 28
    i_height = 28
    my_image = scipy.misc.imresize(image, (i_height, i_width))  
#    my_image = resize(image, (i_height, i_width))    

    #### convert into gray image
    gray_image = rgb2gray(my_image)    
    gray_image = gray_image[..., np.newaxis]  
    return gray_image

test_image = rescale_image(numpy_image)

image = test_image.reshape(1,28,28,1)

###########################################################################
### Predict from file
###########################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
from propagation_forward import *
  
def predict_from_restored_session(X):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    
    x, y, lr, pkeep, step = create_placeholders(28, 28, 1, 10)
    parameters = initialize_parameters()

    AL, _, cnnL1, cnnL2, cnnL3, dense = forward_propagation_with_layers(X, parameters, pkeep)
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()        
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
           
        # Run the initialization
        sess.run(init)

        # Restore variables from disk.
        saver.restore(sess, model_path)
        
        Y_hat = sess.run(AL, feed_dict = {x: X, pkeep: 1.0})
        
        L1 = sess.run(cnnL1, feed_dict = {x: X, pkeep: 1.0})
        L2 = sess.run(cnnL2, feed_dict = {x: X, pkeep: 1.0})
        L3 = sess.run(cnnL3, feed_dict = {x: X, pkeep: 1.0})
        D1 = sess.run(dense, feed_dict = {x: X, pkeep: 1.0})
                        
        return Y_hat, L1, L2, L3, D1
    

pred_from_sess, L1, L2, L3, D1 = predict_from_restored_session(image)



###########################################################################
### Rest of the code below is just for exporting all layers into NodeJS
###########################################################################

# CNN Layer 1
layer1_image1 = L1[0,:,:,0]
layer1_image2 = L1[0,:,:,1]
layer1_image3 = L1[0,:,:,2]
layer1_image4 = L1[0,:,:,3]
layer1_image5 = L1[0,:,:,4]
layer1_image6 = L1[0,:,:,5]
cnn_l1_arr = [layer1_image1, 
              layer1_image2, 
              layer1_image3, 
              layer1_image4, 
              layer1_image5, 
              layer1_image6]

# CNN Layer 2
layer2_image1 = L2[0,:,:,0]
layer2_image2 = L2[0,:,:,1]
layer2_image3 = L2[0,:,:,2]
layer2_image4 = L2[0,:,:,3]
layer2_image5 = L2[0,:,:,4]
layer2_image6 = L2[0,:,:,5]
layer2_image7 = L2[0,:,:,6]
layer2_image8 = L2[0,:,:,7]
layer2_image9 = L2[0,:,:,8]
layer2_image10 = L2[0,:,:,9]
layer2_image11 = L2[0,:,:,10]
layer2_image12 = L2[0,:,:,11]
cnn_l2_arr = [layer2_image1, 
              layer2_image2, 
              layer2_image3, 
              layer2_image4, 
              layer2_image5, 
              layer2_image6,
              layer2_image7,
              layer2_image8,
              layer2_image9,
              layer2_image10,
              layer2_image11,
              layer2_image12]



# CNN Layer 3
layer3_image1 = L3[0,:,:,0] # Display it through array loop

# DENSE Layer
dense_vis = D1[0,:]

xmax, xmin = pred_from_sess.max(), pred_from_sess.min()
pred_normalized = (pred_from_sess - xmin)/(xmax - xmin)

####################################
# Show layers
import matplotlib.pyplot as plt

def display_cnn_layer_one():
    fig=plt.figure(figsize=(8, 8))
    columns = len(cnn_l1_arr)
    rows = 1
    for i in range(1, columns+1):
    #    img = np.random.randint(10, size=(h,w))
        ind = i-1
        img = cnn_l1_arr[ind]
        fig.add_subplot(rows, columns, i)
        plt.title("Image: " + str(ind))
        plt.imshow(img)
    plt.show()
    
def display_cnn_layer_two():
    fig=plt.figure(figsize=(8, 8))
    columns = len(cnn_l2_arr)
    rows = 1
    for i in range(1, columns+1):
    #    img = np.random.randint(10, size=(h,w))
        ind = i-1
        img = cnn_l2_arr[ind]
        fig.add_subplot(rows, columns, i)
        plt.title("Image: " + str(ind))
        plt.imshow(img)
    plt.show()

def display_cnn_layer_three():
    layer3 = L3[0,:,:,:] # shape (7, 7, 24)
    images_count = L3[0,:,:,:].shape[2] # 24
    
    fig=plt.figure(figsize=(8, 8))
    columns = images_count
    rows = 1
    for i in range(1, images_count+1):
    #    img = np.random.randint(10, size=(h,w))
        ind = i-1
        img = layer3[:,:,ind]
        fig.add_subplot(rows, columns, i)
        plt.title("Image: " + str(ind))
        plt.imshow(img)
    plt.show()

def to_base64(numpy_image):
    """
    Method converts numpy image into base64 string
    """

    pil_img = Image.fromarray(numpy_image)
    pil_img = pil_img.convert('RGB')
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def pack(image_b64, layer_num, image_num):
     return "base64_layer"+str(layer_num)+"_image"+str(image_num)+"--->"+str(image_b64)+"<---base64_layer"+str(layer_num)+"_image"+str(image_num)+"_"

## CNN LAYER 1
output_layer1 = ""
for i in range(0,6):
    output_layer1 += pack(to_base64(L1[0,:,:,i]),layer_num=1,image_num=i+1)

## CNN LAYER 2
output_layer2 = ""
for i in range(0,12):
    output_layer2 += pack(to_base64(L2[0,:,:,i]),layer_num=2,image_num=i+1)

## CNN LAYER 3
output_layer3 = ""
for i in range(0,24):
    output_layer3 += pack(to_base64(L3[0,:,:,i]),layer_num=3,image_num=i+1)


# DENSE Layer
dense_layer = D1[0,:]
xmax, xmin = dense_layer.max(), dense_layer.min()
dense_layer_normalized = (dense_layer - xmin)/(xmax - xmin)

dense_incices_non_zero = np.nonzero(dense_layer_normalized)
dense_json = "{"
for idx, val in enumerate(dense_incices_non_zero[0]):
    dense_value = "\"index"+ str(idx) + "\":\"" + str(val) + "tuple" + str(dense_layer_normalized[val]) + str("\",")
    dense_json += dense_value
dense_json = dense_json[:-1]
dense_json += "}"

output_dense = "dense_layer--->" + str(dense_json) + "<---dense_layer"

# PREDICTION
predicted_num = "Predicted_number--->" + str(np.argmax(pred_normalized)) + "<---Predicted_number"

print(output_layer1 + 
      output_layer2 +
      output_layer3 +
      output_dense + 
      predicted_num
      )
sys.stdout.flush()
