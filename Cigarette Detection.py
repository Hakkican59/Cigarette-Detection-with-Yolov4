
import cv2
import numpy as np
import pyttsx3

img=cv2.imread("D:/Desktop/Python/kodlar/yolo/kod/images/smoking_0783.jpg")
img_width=img.shape[1]
img_height=img.shape[0]
img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)

#Object that we want to detect
labels = ["cigarette"]

#desired color for bounding boxes
colors=["0,255,0"]
colors=[np.array(color.split(",")).astype("int")for color in colors]
colors=np.array(colors)
colors=np.tile(colors,(15,1))

#cfg and trained weights file
model=cv2.dnn.readNetFromDarknet("D:\Desktop\Python\kodlar\yolo\my_model\spot_yolov4.cfg","D:\Desktop\Python\kodlar\yolo\my_model\spot_yolov4_final.weights")
layers=model.getLayerNames()

output_Layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]
model.setInput(img_blob)
detection_layers=model.forward(output_Layer)

catch= False
engine = pyttsx3.init()

ids_list=[]
boxes_list=[]
confidence_list=[]

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores=object_detection[5:]
        predicted_id=np.argmax(scores)
        confidence=scores[predicted_id]
        ##accuracy percentige
        if confidence>0.40:
            catch=True
            label=labels[predicted_id]
            bounding_box=object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
            box_centerx,box_centery,box_width,box_height=bounding_box.astype("int")
            startx=int(box_centerx-(box_width/2))
            starty=int(box_centery-(box_height/2))
            

            ids_list.append(predicted_id)
            confidence_list.append(float(confidence))
            boxes_list.append([startx,starty,int(box_width),int(box_height)])

           
maxids=cv2.dnn.NMSBoxes(boxes_list,confidence_list,0.5,0.4)
for maxid in maxids:
    max_class_id = maxid
    box=boxes_list[max_class_id]
    
    startx=box[0]
    starty=box[1]
    box_width=box[2]
    box_height=box[3]   

    predicted_id=ids_list[max_class_id]
    label=labels[predicted_id]
    confidence=confidence_list[max_class_id]  


    endx=startx+box_width
    endy=starty+box_height
    
    box_color=colors[predicted_id]
    box_color=[int(each)for each in box_color]
    
    label="{}:{:.2f}%".format(label,confidence*100)
    print("predicted object {}".format(label))
    
    cv2.rectangle(img,(startx,starty),(endx,endy),box_color,1)
    cv2.putText(img,label,(startx,starty-10),cv2.FONT_HERSHEY_COMPLEX,0.5,box_color,1)        
            
cv2.imshow("Detection Window",img)
if catch :
    engine.say("Cigarette detected!")
    engine.runAndWait()
   

