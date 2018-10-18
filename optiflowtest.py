import cv2
import numpy as np

max_u = 0
max_v = 0
min_u = 1000
min_v = 1000
pt_idx = 0
img_pts0 = np.zeros((3, 4), dtype=np.float32)
img_pts1 = np.zeros((3, 4), dtype=np.float32)


def mouse_calback(event, x, y, flags, param):
    global max_u, max_v, min_u, min_v, pt_idx, image
    if event == cv2.EVENT_LBUTTONDOWN and pt_idx < 4:
        print(pt_idx)
        print('lbtd:'+str(x)+','+str(y))
        img_pts0[0, pt_idx] = x
        img_pts0[1, pt_idx] = y
        img_pts0[2, pt_idx] = 1
        if x > max_u:
            max_u = x
        if x < min_u:
            min_u = x
        if y > max_v:
            max_v = y
        if y < min_v:
            min_v = y
        print(img_pts0)
        print(str(min_u)+','+str(min_v)+','+str(max_u)+','+str(max_v))
        pt_idx += 1
    # if event == cv2.EVENT_LBUTTONDOWN and pt_idx == 4:
    #     #cv2.rectangle(image,(min_u,min_v),(max_u,max_v),(0,255,0))
    #     cv2.imshow('image', image)


image = cv2.imread('/home/ryan/ads_implant/jpg/01168.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_calback)
cv2.imshow('image', image)
cv2.waitKey(0)

lastimg=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

idx1 = 0
for idx in range(1168, 1280):
    fn_img = "/home/ryan/ads_implant/jpg/%05d.jpg" % idx
    #fn_img_1 = "new_%05d.jpg" % idx1
    image2 = cv2.imread(fn_img)

    min_u, min_v = np.min(img_pts0[0:2, :], 1)
    max_u, max_v = np.max(img_pts0[0:2, :], 1)
    #rectimg = np.zeros(image.shape[0:2], dtype = "uint8")
    cv2.rectangle(image2, (int(min_u), int(min_v)),
                  (int(max_u), int(max_v)), (255,0,0))


    currimg=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(lastimg, currimg, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # draw lines 
    step=10
    h, w = currimg.shape[:2]
    print(currimg.shape[:2])
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int) 
    print(y)
    print(x)
    fx, fy = flow[y, x].T 
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2) 
    lines = np.int32(lines) 
    line = [] 
    for l in lines: 
        if l[0][0]-l[1][0]>0 or l[0][1]-l[1][1]>0: 
            line.append(l) 
    cv2.polylines(image2, line, 0, (0,255,255))

    cv2.imshow('image2', image2)

    # cv2.imshow('lastimg',lastimg)
    # cv2.imshow('currimg',currimg)

    # subimg=currimg-lastimg
    # cv2.imshow('subimg',subimg)

    cv2.waitKey(0)

    lastimg=currimg.copy()
