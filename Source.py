import cv2
import numpy as np
import matplotlib.pyplot as plt
import tao_asari
import WLS_filter
import histogram_process as hp
import face_detector

print ("Enter image path")
#img_path = raw_input()
img_path = '.\Images\dsc_0052.jpg'

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

list_of_histogram_doublets =[]
for (x,y,w,h) in faces:
    skinMask = face_detector.detect_skin(orig_img[y:y+h,x:x+w])
    face_portion = base_layer[y:y+h,x:x+w]

    # cv2.imshow("Original",cv2.bitwise_and(face_portion,face_portion, mask=skinMask))
    # cv2.waitKey(0)

    hist, bins = hp.generate_histogram(cv2.bitwise_and(face_portion,face_portion, mask=skinMask))
    #hist, bins = hp.generate_histogram(face_portion)
    plt.figure()
    plt.plot(bins,hist)

    doublet= [hist,bins]
    list_of_histogram_doublets.append(doublet)

for i in range(len(list_of_histogram_doublets)):
    doublet = list_of_histogram_doublets[i]
    d,m,b = hp.detect_modes(doublet[0],doublet[1],0.05)
    # for elem in doublet[0]:
    #     print (str(elem)+','),
    # print '\n'
    # for elem in doublet[1]:
    #     print (str(elem)+','),
    print (d,m,b)
    face = faces[i]
    face_image = base_layer[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
    hp.enhance_sidelit_face(face_image,doublet,d,m,b)


    #cv2.imshow("ab"+str(i),face_image)



# cv2.imshow("Original",orig_hsv_image)
# cv2.imshow("Base",base_layer)
# cv2.imshow("Detail",detail_layer)
plt.show()
cv2.waitKey(0)