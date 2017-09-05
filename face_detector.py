import os,math
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

def apply_threshold(value):
    return 255 * math.floor(value / 128)


def floyd_steinberg_dither(image):  # 2d array
    pixel = np.copy(image)
    pixel = pixel*255
    y_lim = image.shape[1]
    x_lim = image.shape[0]
    for y in range(1, y_lim):
        for x in range(1, x_lim):
            oldpixel = pixel[x, y]
            newpixel = apply_threshold(oldpixel)
            pixel[x, y] = newpixel
            error = oldpixel - newpixel

            if x < x_lim - 1:
                red = pixel[x + 1, y] + round(error * 7.0 / 16)
                pixel[x + 1, y] = red

            if x > 1 and y < y_lim - 1:
                red = pixel[x - 1, y + 1] + round(error * 3.0 / 16)
                pixel[x - 1, y + 1] = red

            if y < y_lim - 1:
                red = pixel[x, y + 1] + round(error * 5.0 / 16)
                pixel[x, y + 1] = red

            if x < x_lim - 1 and y < y_lim - 1:
                red = pixel[x + 1, y + 1] + round(error * 1.0 / 16)
                pixel[x + 1, y + 1] = red

    return pixel/255.0

def floyd_steinberg_dither_3Channel(image): # 2d array
    pixel = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    y_lim = image.shape[1]
    x_lim = image.shape[0]
    for y in range(1, y_lim):
        for x in range(1, x_lim):
            red_oldpixel, green_oldpixel, blue_oldpixel = pixel[x, y]

            red_newpixel = apply_threshold(red_oldpixel)
            green_newpixel = apply_threshold(green_oldpixel)
            blue_newpixel = apply_threshold(blue_oldpixel)

            pixel[x, y] = red_newpixel, green_newpixel, blue_newpixel

            red_error = red_oldpixel - red_newpixel
            blue_error = blue_oldpixel - blue_newpixel
            green_error = green_oldpixel - green_newpixel

            if x < x_lim - 1:
                red = pixel[x+1, y][0] + round(red_error * 7/16)
                green = pixel[x+1, y][1] + round(green_error * 7/16)
                blue = pixel[x+1, y][2] + round(blue_error * 7/16)

                pixel[x+1, y] = (red, green, blue)

            if x > 1 and y < y_lim - 1:
                red = pixel[x-1, y+1][0] + round(red_error * 3/16)
                green = pixel[x-1, y+1][1] + round(green_error * 3/16)
                blue = pixel[x-1, y+1][2] + round(blue_error * 3/16)

                pixel[x-1, y+1] = (red, green, blue)

            if y < y_lim - 1:
                red = pixel[x, y+1][0] + round(red_error * 5/16)
                green = pixel[x, y+1][1] + round(green_error * 5/16)
                blue = pixel[x, y+1][2] + round(blue_error * 5/16)

                pixel[x, y+1] = (red, green, blue)

            if x < x_lim - 1 and y < y_lim - 1:
                red = pixel[x+1, y+1][0] + round(red_error * 1/16)
                green = pixel[x+1, y+1][1] + round(green_error * 1/16)
                blue = pixel[x+1, y+1][2] + round(blue_error * 1/16)

                pixel[x+1, y+1] = (red, green, blue)

    return cv2.cvtColor(pixel,cv2.COLOR_RGB2BGR)

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