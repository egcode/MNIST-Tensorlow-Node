import sys
import os
import numpy as np
from PIL import Image
from scipy import ndimage
import scipy
#from skimage.transform import resize

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable Deprecation Warnings
import warnings
warnings.filterwarnings("ignore")



#model_path = '/Users/eugene/Desktop/nodeMNIST/Python_NN/saved_model/model.ckpt'
#raw_image_data_string = open('test_image.txt', 'r').read() # FROM FILE
 
raw_image_data_string = sys.argv[1]    # FROM NODE
model_path = sys.argv[2]               # FROM NODE




# clear base64 image data
string_to_remove = "data:image/jpeg;base64,"
image_base_64_string = raw_image_data_string.replace(string_to_remove, "")
#print("imagedata: " + str(image_base_64_string))


import base64
import io
#from matplotlib import pyplot as plt
import matplotlib.image as mpimg

img = base64.b64decode(image_base_64_string)
img = io.BytesIO(img)
numpy_image = mpimg.imread(img, format='JPG')

#plt.imshow(numpy_image, interpolation='nearest')
#plt.show()


#print("image shape: " + str(numpy_image.shape))

######################################################

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
#plt.imshow(test_image.squeeze(), cmap='gray')

image = test_image.reshape(1,28,28,1)
dummy_label = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
#print("NEW image shape: " + str(image.shape))

#pred_from_sess = predict_from_restored_session(image, dummy_label)
#print("Prediction: " + str(pred_from_sess))
#xmax, xmin = pred_from_sess.max(), pred_from_sess.min()
#pred_normalized = (pred_from_sess - xmin)/(xmax - xmin)
#print("After normalization:")
#print(pred_normalized)
#print("\n")
#print("Predicted number   --->   " + str(np.argmax(pred_normalized)))



import tensorflow as tf
from tensorflow.python.framework import ops
from propagation_forward import *

    
def predict_from_restored_session(X):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    
#    x, y = create_placeholders(28, 28, 1, 10)
#    parameters = initialize_parameters()
#    Z3 = forward_propagation(x, parameters)
    x, y, lr, pkeep, step = create_placeholders(28, 28, 1, 10)
    parameters = initialize_parameters()
#    AL, _ = forward_propagation(X, parameters, pkeep)
    AL, _, cnnL1, cnnL2, cnnL3, dense = forward_propagation_with_layers(X, parameters, pkeep)

    forward_propagation_with_layers
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()        
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
           
        # Run the initialization
        sess.run(init)

        # Restore variables from disk.
#        saver.restore(sess, "./saved_model/model.ckpt")
        saver.restore(sess, model_path)
#        print("Model restored.")
        
        Y_hat = sess.run(AL, feed_dict = {x: X, pkeep: 1.0})
        
        L1 = sess.run(cnnL1, feed_dict = {x: X, pkeep: 1.0})
        L2 = sess.run(cnnL2, feed_dict = {x: X, pkeep: 1.0})
        L3 = sess.run(cnnL3, feed_dict = {x: X, pkeep: 1.0})
        D1 = sess.run(dense, feed_dict = {x: X, pkeep: 1.0})

#        predict_op = tf.argmax(Y_hat, 1)
                        
        return Y_hat, L1, L2, L3, D1
    
###############
### Predict from file
##############

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


#test_image = get_image("./mnist_test_images/3.png")
#plt.imshow(test_image.squeeze(), cmap='gray')

#image = test_image.reshape(1,28,28,1)
#dummy_label = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


#pred_from_sess = predict_from_restored_session(image, dummy_label)
pred_from_sess, L1, L2, L3, D1 = predict_from_restored_session(image)

#print("Prediction: " + str(pred_from_sess))
#print("Layer1 with 6 28x28 images shape: " + str(L1.shape))
#print("Layer2 with 12 14x14 images shape: " + str(L2.shape))
#print("Layer3 with 24 7x7 images shape: " + str(L3.shape))
#print("Dense Layer with 200 neurons shape: " + str(D1.shape))

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
#print("After normalization:")
#print(pred_normalized)

#print("\n")
#print("Predicted number--->" + str(np.argmax(pred_normalized)) + "<---Predicted number")
#print("\n")

sys.stdout.flush()





####################################
# Show layers
import matplotlib.pyplot as plt

#plt.title("Layer 1 image 1")
#plt.imshow(layer1_image1, cmap='gray')
#
#plt.imshow(layer1_image2, cmap='gray')
#
#

#w=10
#h=10
#fig=plt.figure(figsize=(8, 8))
#columns = 6
#rows = 5
#for i in range(len(l1_arr)):
#    fig.add_subplot(rows, columns, i)
#    plt.imshow(l1_arr[i], cmap='gray')
#plt.show()

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

    

#display_cnn_layer_one()
#display_cnn_layer_two()
#display_cnn_layer_three()

#plt.title("Layer 1 image 1")
#plt.imshow(layer1_image1, cmap='gray')

#plt.title("Layer 2 image 1")
#plt.imshow(layer2_image1, cmap='gray')
#
#plt.title("Layer 3 image 1")
#plt.imshow(layer3_image1, cmap='gray')

#plt.title("Dense 1")
#plt.imshow(dense_vis, cmap='gray')

#for i in range(200):
#    print(D1[0,i])


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


#base64_layer1_image1 = "base64_layer1_image1--->" + str(to_base64(layer1_image1)) + "<---base64_layer1_image1"
#base64_layer1_image2 = "base64_layer1_image2--->" + str(to_base64(layer1_image2)) + "<---base64_layer1_image2"
#base64_layer1_image3 = "base64_layer1_image3--->" + str(to_base64(layer1_image3)) + "<---base64_layer1_image3"
#base64_layer1_image4 = "base64_layer1_image4--->" + str(to_base64(layer1_image4)) + "<---base64_layer1_image4"
#base64_layer1_image5 = "base64_layer1_image5--->" + str(to_base64(layer1_image5)) + "<---base64_layer1_image5"
#base64_layer1_image6 = "base64_layer1_image6--->" + str(to_base64(layer1_image6)) + "<---base64_layer1_image6"

## LAYER 1
#base64_layer1_image1 = pack(to_base64(L1[0,:,:,0]),layer_num=1,image_num=1)
#base64_layer1_image2 = pack(to_base64(L1[0,:,:,1]),layer_num=1,image_num=2)
#base64_layer1_image3 = pack(to_base64(L1[0,:,:,2]),layer_num=1,image_num=3)
#base64_layer1_image4 = pack(to_base64(L1[0,:,:,3]),layer_num=1,image_num=4)
#base64_layer1_image5 = pack(to_base64(L1[0,:,:,4]),layer_num=1,image_num=5)
#base64_layer1_image6 = pack(to_base64(L1[0,:,:,5]),layer_num=1,image_num=6)

## LAYER 2
#base64_layer2_image1 = pack(to_base64(L2[0,:,:,0]),layer_num=2,image_num=1)
#base64_layer2_image2 = pack(to_base64(L2[0,:,:,1]),layer_num=2,image_num=2)
#base64_layer2_image3 = pack(to_base64(L2[0,:,:,2]),layer_num=2,image_num=3)
#base64_layer2_image4 = pack(to_base64(L2[0,:,:,3]),layer_num=2,image_num=4)
#base64_layer2_image5 = pack(to_base64(L2[0,:,:,4]),layer_num=2,image_num=5)
#base64_layer2_image6 = pack(to_base64(L2[0,:,:,5]),layer_num=2,image_num=6)
#base64_layer2_image7 = pack(to_base64(L2[0,:,:,6]),layer_num=2,image_num=7)
#base64_layer2_image8 = pack(to_base64(L2[0,:,:,7]),layer_num=2,image_num=8)
#base64_layer2_image9 = pack(to_base64(L2[0,:,:,8]),layer_num=2,image_num=9)
#base64_layer2_image10 = pack(to_base64(L2[0,:,:,9]),layer_num=2,image_num=10)
#base64_layer2_image11 = pack(to_base64(L2[0,:,:,10]),layer_num=2,image_num=11)
#base64_layer2_image12 = pack(to_base64(L2[0,:,:,11]),layer_num=2,image_num=12)


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


#print("\n")
#print("base64_layer1_image1--->" + str(base64_layer1_image1) + "<---base64_layer1_image1")
#print("\n")
#print("base64_layer1_image2--->" + str(base64_layer1_image2) + "<---base64_layer1_image2")
#print("\n")
#print("base64_layer1_image3--->" + str(base64_layer1_image3) + "<---base64_layer1_image3")
#print("\n")
#print("base64_layer1_image4--->" + str(base64_layer1_image4) + "<---base64_layer1_image4")
#print("\n")
#print("base64_layer1_image5--->" + str(base64_layer1_image5) + "<---base64_layer1_image5")
#print("\n")
#print("base64_layer1_image6--->" + str(base64_layer1_image6) + "<---base64_layer1_image6")
#print("\n")

#print("\n")
#print(predicted_num + 
#      base64_layer1_image1 + 
#      base64_layer1_image2 + 
#      base64_layer1_image3 + 
#      base64_layer1_image4 + 
#      base64_layer1_image5 + 
#      base64_layer1_image6 +
#      output_layer2
      
#      base64_layer2_image1 +
#      base64_layer2_image2 +
#      base64_layer2_image3 +
#      base64_layer2_image4 +
#      base64_layer2_image5 +
#      base64_layer2_image6 +
#      base64_layer2_image7 +
#      base64_layer2_image8 +
#      base64_layer2_image9 +
#      base64_layer2_image10 +
#      base64_layer2_image11 +
#      base64_layer2_image12
#      )

print(output_layer1 + 
      output_layer2 +
      output_layer3 +
      output_dense + 
      predicted_num
      )

#print("\n")
