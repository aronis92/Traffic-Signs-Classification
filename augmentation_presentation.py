from pandas.io.parsers import read_csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from IPython.display import display


img = Image.open('./images/demo.jpg').resize((400,400), Image.NEAREST)
np_img = np.array(img)
# Bilateral filtering
img2 = cv2.bilateralFilter(np_img,9,75,75)
img2 = Image.fromarray(img2)
# Gaussian Filtering
img3 = cv2.GaussianBlur(np_img,(5,5),0)
img3 = Image.fromarray(img3)
# perspective transformation
pts1 = np.float32([[30,30],[370,30],[30,370],[370,370]])
pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
M = cv2.getPerspectiveTransform(pts1,pts2)
img4 = cv2.warpPerspective(np_img,M,(400,400))
img4 = Image.fromarray(img4)
# rotation + 10
image_shape = np_img.shape[1::-1]
image_center = tuple(np.array(image_shape)/2)
rot_mat = cv2.getRotationMatrix2D(image_center, 10, 1.0)
img5 = cv2.warpAffine(np_img, rot_mat, np_img.shape[1::-1], flags=cv2.INTER_LINEAR)
img5 = Image.fromarray(img5)
# rotation -10
rot_mat = cv2.getRotationMatrix2D(image_center, -10, 1.0)
img6 = cv2.warpAffine(np_img, rot_mat, np_img.shape[1::-1], flags=cv2.INTER_LINEAR)
img6 = Image.fromarray(img6)
# translation
M = np.float32([[1,0,20],[0,1,20]])
img7 = cv2.warpAffine(np_img,M,(400,400))
img7 = Image.fromarray(img7)

image_list = [img2, img3, img4, img5, img6, img7]
w, h, _ = np_img.shape
X = np.ones((400,20,3), dtype='uint8')
X = X*255
X = Image.fromarray(X)
for i in range(6):
    offset = 0
    new_im = Image.new('RGB', (w*2 + 20, h))
    new_im.paste(img, (offset,0))
    offset += w
    new_im.paste(X, (offset,0))
    offset += 20
    new_im.paste(image_list[i], (offset,0))
    new_im.save('./images/preproc'+str(i+1)+'.png')

pts1 = np.float32([[30,30],[370,30],[30,370],[370,370]])
pts2 = np.float32([[0,0],[400,0],[0,400],[400,400]])
M = cv2.getPerspectiveTransform(pts1,pts2)
img8 = cv2.warpPerspective(np_img,M,(400,400)) 
image_shape = img8.shape[1::-1]
image_center = tuple(np.array(image_shape)/2)
rot_mat = cv2.getRotationMatrix2D(image_center, 10, 1.0)
img8 = cv2.warpAffine(img8, rot_mat, np_img.shape[1::-1], flags=cv2.INTER_LINEAR)
img8 = Image.fromarray(img8)

offset = 0
new_im = Image.new('RGB', (w*4 + 60, h))
new_im.paste(img, (offset,0))
offset += w
new_im.paste(X, (offset,0))
offset += 20
new_im.paste(image_list[2], (offset,0))
offset += w
new_im.paste(X, (offset,0))
offset += 20
new_im.paste(image_list[3], (offset,0))
offset += w
new_im.paste(X, (offset,0))
offset += 20
new_im.paste(img8, (offset,0))
new_im.save('./images/filter_on_filter.png')

