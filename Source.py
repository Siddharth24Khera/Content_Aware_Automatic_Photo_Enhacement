import cv2
import numpy as np
import matplotlib.pyplot as plt
import tao_asari
import WLS_filter
import histogram_process as hp
import face_detector
import sky_enhancement

#print ("Enter image path")
#img_path = raw_input()
img_path = '.\Images\und.jpg'

orig_img = cv2.imread(img_path)

# Face Detection
tao_asari_enhanced_img = tao_asari.tao_asari_enhancement(orig_img)
faces = face_detector.face_detect(tao_asari_enhanced_img)
rect_faces = np.copy(orig_img)
for (x, y, w, h) in faces:
    cv2.rectangle(rect_faces, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow("detected_faces",rect_faces)
orig_hsv_image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)

base_layer, detail_layer = WLS_filter.wls_filter(orig_hsv_image[:,:,2])

list_of_histogram_triplets =[]
for (x,y,w,h) in faces:
    skinMask = face_detector.detect_skin(tao_asari_enhanced_img[y:y+h,x:x+w])
    face_portion = base_layer[y:y+h,x:x+w]

    hist, bins = hp.generate_histogram(cv2.bitwise_and(face_portion,face_portion, mask=skinMask))
    #hist, bins = hp.generate_histogram(face_portion)
    plt.figure()
    plt.plot(bins,hist)

    triplet= [hist,bins,skinMask]
    list_of_histogram_triplets.append(triplet)

for i in range(len(list_of_histogram_triplets)):
    triplet = list_of_histogram_triplets[i]
    face = faces[i]
    face_image = base_layer[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]

    base_layer[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = hp.enhance_sidelit_face(face_image,triplet)
    base_layer[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = hp.enhance_underexposed(face_image, triplet)


orig_hsv_image[:,:,2] = (base_layer+detail_layer)*255
final_image = cv2.cvtColor(orig_hsv_image, cv2.COLOR_HSV2BGR)


for (x, y, w, h) in faces:
    # final_image[y:y + h, x:x + w] = face_detector.floyd_steinberg_dither_3Channel(final_image[y:y + h, x:x + w])
    final_image[y:y+h,x:x+w]=cv2.bilateralFilter(final_image[y:y+h,x:x+w],5,300,300)

final_image,_ = sky_enhancement.sky_enhancement(final_image)
im = np.hstack((orig_img,final_image))

cv2.imshow("Original", im)
plt.show()
key = cv2.waitKey(20)
if key & 0xFF == ord('s'):
    cv2.imwrite('output.jpg', im)

cv2.destroyAllWindows()
plt.close()