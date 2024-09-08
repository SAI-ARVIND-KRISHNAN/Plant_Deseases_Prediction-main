#Library imports
import numpy as np
#import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model("/Users/saiarvind/Desktop/Plant_Deseases_Prediction-main/Plant_Deseases_Prediction-main\plant_disease.h5")

#Name of Classes
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

#Setting Title of App
#st.title("Plant Disease Detection")


cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, image = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    #cv2.imshow("test", image)
    
    #Resizing the image
    opencv_image = cv2.resize(image, (256,256))
    #Convert image to 4 Dimension
    opencv_image.shape = (1,256,256,3)
    #Make Prediction
    Y_pred = model.predict(opencv_image)
    result = CLASS_NAMES[np.argmax(Y_pred)]

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # org 
    org = (50, 50) 
  
    # fontScale 
    fontScale = 1
   
    # Blue color in BGR 
    color = (255, 0, 0) 
  
    # Line thickness of 2 px 
    thickness = 2
   
    # Using cv2.putText() method 
    image = cv2.putText(image, result, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 


    cv2.imshow("test", image)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break



    '''
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
    '''
    
'''
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray("opencv_frame_{}.jpg".format(img_counter)), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        '''

'''
        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
        
        img_counter += 1
        '''

cam.release()

cv2.destroyAllWindows()

'''
#setting up our cam
cap = cv2.VideoCapture(0)

while True:
    ret, opencv_image = cap.read()

    # Displaying the image
    st.image(opencv_image, channels="BGR")
    st.write(opencv_image.shape)
    #Resizing the image
    opencv_image = cv2.resize(opencv_image, (256,256))
    #Convert image to 4 Dimension
    opencv_image.shape = (1,256,256,3)
    #Make Prediction
    Y_pred = model.predict(opencv_image)
    result = CLASS_NAMES[np.argmax(Y_pred)]

    # Window name in which image is displayed 
    window_name = 'image'
    
    cv2.imshow(window_name, result)
    
    #st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
cap.release()
'''    

'''
st.markdown("Upload an image of the plant leaf")

#Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
#On predict button click
if submit:


    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("This is "+result.split('-')[0]+ " leaf with " + result.split('-')[1]))
'''
