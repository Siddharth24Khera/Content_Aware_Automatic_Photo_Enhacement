import os
import cv2
import tao_asari
import numpy as np

def face_detect(tao_asari_enhanced_img):
    face_cascade = cv2.CascadeClassifier('C:\Users\Siddharth Khera\Desktop\opencv_3.0\sources\data\haarcascades_cuda' +
                                         '\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(tao_asari_enhanced_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(30, 30)
    )

    return faces


def detect_skin(enhaced_img_face):
    lower = np.array([0, 20, 0], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    orig_img = enhaced_img_face
    orig_hsv_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(orig_hsv_image, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    # print skinMask.shape
    # skin = cv2.bitwise_and(orig_img, orig_img, mask=skinMask)

    return skinMask



if __name__ == '__main__':

    IMG_DIR = './Images'
    IMG_NAME = 'group.jpg'
    image_path = os.path.join(IMG_DIR, IMG_NAME)
    img = cv2.imread(image_path)
    tao_asari_enhanced_img = tao_asari.tao_asari_enhancement(img)
    cv2.imshow('as',tao_asari_enhanced_img)
    faces = face_detect(tao_asari_enhanced_img)
    for (x,y,w,h) in faces:
        cv2.rectangle(tao_asari_enhanced_img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('img',tao_asari_enhanced_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()