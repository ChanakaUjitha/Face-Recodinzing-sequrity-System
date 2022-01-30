import numpy as np
from PIL import Image
import os, cv2
import pickle



# Method to train custom classifier to recognize face
def train_classifer(name):
        # Read all the images in custom data-set
        path = os.path.join(os.getcwd()+"/data/"+name+"/")

        faces = []
        ids = []
        labels = []
        pictures = {}


        # Store images in a numpy format and ids of the user on the same index in imageNp and id lists

        for root,dirs,files in os.walk(path):
                pictures = files


        for pic in pictures :

                imgpath = path+pic
                img = Image.open(imgpath).convert('L')
                imageNp = np.array(img, 'uint8')
                id = int(pic.split(name)[0])
                #names[name].append(id)
                faces.append(imageNp)
                ids.append(id)

        ids = np.array(ids)

        #Train and save classifier
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("./data/classifiers/"+name+"_Trainner.yml")
























































        

    #my code################################################################

        BASE_DiR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DiR,"data")

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()


        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []

        for root, dirs, files in os.walk(image_dir):
                for file in files:
                        if file.endswith("png") or file.endswith("jpg"):
                                path = os.path.join(root,file)
                                label = os.path.basename(root).replace(" ","_").lower()
                                #print(label , path)

                                if not label in label_ids:
                                        label_ids[label] = current_id
                                        current_id+=1
                                id_ = label_ids[label]
                                #print(label_ids)
                                #y_labels.append(label)
                                #x_train.append(path)

                                pil_image =Image.open(path).convert("L")
                                size=(512,512)
                                final_image = pil_image.resize(size,Image.ANTIALIAS)
                                image_array = np.array(pil_image,"uint8")
                                # print(image_array)

                                faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

                                for (x,y,w,h) in faces:
                                        roi = image_array[y:y+h,x:x+w]
                                        x_train.append(roi)
                                        y_labels.append(id_)

        with open("labels.pickle",'wb')as f:
                pickle.dump(label_ids,f)

        recognizer.train(x_train,np.array(y_labels))
        recognizer.save("data/train/trainner.yml")
    

