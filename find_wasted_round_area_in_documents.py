# zeinab Taghavi
#
# its better image be threshed once before usage
# 1 - we need dpi for slicing image
# 2 - use erod nad then dilate in order to clear small noises
# 3 - by semi-histogram way we want to find wasted areas
#


import cv2
import numpy as np
from lxml import etree
import pytesseract
from PIL import Image

def find_wasted_round_area_in_documents(img_file):

    img = cv2.imread(img_file)
    cv2.imwrite('image1.jpg',img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1 - we need dpi for slicing image
    imgPIL = Image.open(img_file)
    dpi = (300, 300)  # default is (300 , 300)
    if 'dpi' in imgPIL.info.keys():
        dpi = imgPIL.info['dpi']
    del imgPIL

    # 2 - use erod nad then dilate in order to clear small noises
    gray_env = cv2.bitwise_not(gray)
    kernel_erod = np.ones((10,10),np.uint8)
    gray_env_erod = cv2.erode(gray_env , kernel_erod , iterations=1)

    kernel_dilate = np.ones((15,15),np.uint8)
    gray_env_dilate = cv2.dilate(gray_env_erod , kernel_dilate , iterations=2)

    # 3 - by semi-histogram way we want to find wasted areas

    slice = int(dpi[0]/20)
    cv2.imwrite('find_wasted_round_area_in_documents_1_inv.jpg', gray_env_dilate)

    poly = np.zeros((int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice), 1), np.uint8)
    poly.fill(0)
    pices = (int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice))
    for y in range(pices[0]):
        for x in range(pices[1]):
            poly[y, x] = np.mean(gray_env_dilate[(y * slice):((y + 1) * slice), (x * slice):((x + 1) * slice)])
    _, poly = cv2.threshold(poly, 10, 255, cv2.THRESH_BINARY)
    cv2.imwrite('find_wasted_round_area_in_documents_2_poly_1.jpg', poly)

    poly[0:5, :] = 255
    poly[poly.shape[0] - 5: poly.shape[0], :] = 255
    poly[:, 0:5] = 255
    poly[:, poly.shape[1] - 5:poly.shape[1]] = 255
    h, w = poly.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(poly, mask, (0, 0), 0)

    cv2.imwrite('find_wasted_round_area_in_documents_3_poly_2_floodfill.jpg', poly)
    poly2 = np.zeros((int(gray_env_dilate.shape[0] / slice), int(gray_env_dilate.shape[1] / slice), 1), np.uint8)
    poly2.fill(0)
    for y in range(2, pices[0] - 2):
        for x in range(2, pices[1] - 2):
            if (np.mean(poly[y - 2:y + 3, x - 2:x + 3]) > 20):
                poly2[y - 2:y + 3, x - 2:x + 3] = 255
            else:
                poly2[y, x] = 0

    cv2.imwrite('find_wasted_round_area_in_documents_4_poly2.jpg', poly2)


    del poly
    poly3 = np.zeros((int(gray_env_dilate.shape[0]), int(gray_env_dilate.shape[1]), 1), np.uint8)
    poly3.fill(0)
    for y in range(0, pices[0]):
        for x in range(0, pices[1]):
            poly3[(y * slice):((y + 1) * slice), (x * slice):((x + 1) * slice)] = poly2[y, x]

    cv2.imwrite('find_wasted_round_area_in_documents_5_poly3.jpg', poly3)
    del poly2

    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    no_waisted_area = cv2.bitwise_not(poly3)
    no_waisted_area_on_source = cv2.bitwise_or(gray_img, no_waisted_area)

    cv2.imwrite('find_wasted_round_area_in_documents_6_no_waisted_area.jpg', no_waisted_area_on_source)

if __name__ == '__main__':
    n1 = 1
    n2 = 2
    for i in range(n1, n2):
        img_file = 'image'+str(i)+'.jpg'
        find_wasted_round_area_in_documents(img_file)
