import cv2
import numpy as np
import math

# Reading,displaying and saving images
# Color spaces
# Image translation
# Image rotation
# Image scaling
# Affine transformation
# Projective transformation(homography)
# Image warping

img = cv2.imread("/home/kuluruvineeth/Pictures/tesla.jpeg")
print(img.shape)
gray_img=cv2.imread("/home/kuluruvineeth/Pictures/tesla.jpeg",cv2.IMREAD_GRAYSCALE)
gray1_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
yuv_img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow("Input image",img)
cv2.imshow("Grayscale",gray_img)
cv2.imshow("Grayscale image",gray1_img)
cv2.imshow("YUV image",yuv_img)
cv2.imshow('Y channel',yuv_img[:,:,0])
cv2.imshow('U channel',yuv_img[:,:,1])
cv2.imshow('V channel',yuv_img[:,:,2])
cv2.imshow("HSV image",hsv_img)
cv2.imshow("H channel",hsv_img[:,:,0])
cv2.imshow("S channel",hsv_img[:,:,1])
cv2.imshow("V channel",hsv_img[:,:,2])
cv2.imwrite("/home/kuluruvineeth/Pictures/tesla_grey.jpeg",gray_img)
cv2.waitKey()

num_rows,num_cols = img.shape[:2]
translation_matrix = np.float32([[1,0,70],[0,1,110]])
img_translation = cv2.warpAffine(img,translation_matrix,(num_cols + 70,num_rows + 110))
cv2.imshow("Translation",img_translation)
cv2.waitKey()
translation_matrix = np.float32([[1,0,70],[0,1,110]])
img_translation = cv2.warpAffine(img,translation_matrix,(num_cols + 70,num_rows + 110))
cv2.imshow("Translation",img_translation)
translation_matrix = np.float32([[1,0,-30],[0,1,-50]])
img_translation = cv2.warpAffine(img,translation_matrix,(num_cols + 70+30,num_rows + 110+50))
cv2.imshow("Translation",img_translation)
cv2.waitKey()
translation_matrix = np.float32([[1,0,int(0.5*num_cols)],[0,1,int(0.5*num_rows)]])
rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows),30,1)
img_rotation = cv2.warpAffine(img,rotation_matrix,(2*num_cols,2*num_rows))
cv2.imshow("Rotation",img_rotation)
cv2.waitKey()

img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation =
cv2.INTER_LINEAR)
cv2.imshow('Scaling - Linear Interpolation', img_scaled)
img_scaled =cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
img_scaled =cv2.resize(img,(450, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size', img_scaled)
cv2.waitKey()

img = cv2.imread("/home/kuluruvineeth/Pictures/tesla.jpeg")
rows, cols = img.shape[:2]
src_points = np.float32([[0,0],[cols-1,0],[0,rows-1]])
dst_points = np.float32([[0,0],[int(0.6*(cols-1)),0],[int(0.4*(cols-1)),rows-1]])
affine_matrix = cv2.getAffineTransform(src_points,dst_points)
img_output = cv2.warpAffine(img,affine_matrix,(cols,rows))
cv2.imshow("Input",img)
cv2.imshow("Output",img_output)
cv2.waitKey()
rows, cols = img.shape[:2]
src_points = np.float32([[0,0],[cols-1,0],[0,rows-1]])
dst_points = np.float32([[cols-1,0],[0,0],[cols-1,rows-1]])
affine_matrix = cv2.getAffineTransform(src_points,dst_points)
img_output = cv2.warpAffine(img,affine_matrix,(cols,rows))
cv2.imshow("Input",img)
cv2.imshow("mirror image",img_output)
cv2.waitKey()
rows, cols = img.shape[:2]
src_points = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
dst_points = np.float32([[0,0],[cols-1,0],[int(0.33*cols),rows-1],[int(0.66*cols),rows-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points,dst_points)
img_output = cv2.warpPerspective(img,projective_matrix,(cols,rows))
cv2.imshow("input",img)
cv2.imshow("perspect output",img_output)
cv2.waitKey()

'''
img = cv2.imread("/home/kuluruvineeth/Pictures/tesla.jpeg")
rows = 152
cols = 332

#vertical wave
img_output = np.zeros(img.shape,dtype=img.dtype)

for i in range(rows):
    for j in range(cols):
        offset_x = int(25.0 * math.sin(2 * 3.14 * i/180))
        offset_y = 0
        if j+offset_x < rows:
            img_output[i,j] = img[i,(j+offset_x)%cols]
        else:
            img_output[i,j] = 0
cv2.imshow("imput",img)
cv2.imshow("vertical wave",img_output)
'''
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)
