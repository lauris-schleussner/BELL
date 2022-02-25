# für einen gegebenen Pfad wird ein Pickle Objekt zurückgegeben, wenn rpickle == True gegeben ist (standart)
# sollte rpickle == False wird ein ein Keypoint Objekt und ein Deskriptor Objekt zurückgegeben

import cv2
import pickle
import copyreg

def main(pfad, orb_obj, rpickle = True):
    
    # Bild wird eingelesen und ORB wird ausgeführt
    img = cv2.imread(pfad)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = orb_obj.detectAndCompute(img,None)

    # Pickle ist normalerweise nicht in der Lage Keypoint Objekte zu serialisieren
    # durch copyreg wird eine spezielle Funktion zum serialisieren definiert
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

    if rpickle:
        return pickle.dumps([kp, des])
    else:
        return [kp, des]

# spezielle Funktion um Keypoints zu serialisieren
# https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror/48832618
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)