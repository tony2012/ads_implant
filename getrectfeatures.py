import cv2
import numpy as np


max_u=0
max_v=0
min_u=1000
min_v=1000

pt_idx=0
img_pts0=np.zeros((3,4),dtype=np.float32)

img_pts1=np.zeros((3,4),dtype=np.float32)

def mouse_calback(event, x, y, flags, param):
    global max_u,max_v,min_u,min_v,pt_idx,image
    if event == cv2.EVENT_LBUTTONDOWN and pt_idx<4:
        print(pt_idx)
        print('lbtd:'+str(x)+','+str(y))
        img_pts0[0,pt_idx]=x
        img_pts0[1,pt_idx]=y
        img_pts0[2,pt_idx]=1
        if x>max_u:
            max_u=x
        if x<min_u:
            min_u=x
        if y>max_v:
            max_v=y
        if y<min_v:
            min_v=y
        print(img_pts0)
        print(str(min_u)+','+str(min_v)+','+str(max_u)+','+str(max_v))
        pt_idx+=1
    if event == cv2.EVENT_LBUTTONDOWN and pt_idx==4:


        #cv2.rectangle(image,(min_u,min_v),(max_u,max_v),(0,255,0))
        cv2.imshow('image',image)
        


image=cv2.imread('/Users/zhaopenggu/Projects/pyopencv/jpg/01168.jpg')
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_calback)
cv2.imshow('image',image)
cv2.waitKey(0)

rectimg = np.zeros(image.shape[0:2], dtype = "uint8")
cv2.rectangle(rectimg, (min_u,min_v),(max_u,max_v), 255, -1)
cv2.imshow("rectimg", rectimg)

orb = cv2.ORB_create(10000)
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = orb.detectAndCompute(image,rectimg)
image1=image.copy()
cv2.drawKeypoints(image1,kp1,image1)
cv2.imshow('image1',image1)
cv2.waitKey(0)

idx1=0
for idx in range(1168,1280):
    fn_img="/Users/zhaopenggu/Projects/pyopencv/jpg/%05d.jpg" % idx
    fn_img_1="new_%05d.jpg" % idx1
    image2=cv2.imread(fn_img)

    min_u,min_v=np.min(img_pts0[0:2,:],1)
    max_u,max_v=np.max(img_pts0[0:2,:],1)
    #rectimg = np.zeros(image.shape[0:2], dtype = "uint8")
    cv2.rectangle(rectimg, (int(min_u),int(min_v)),(int(max_u),int(max_v)), 255, -1)
    
    kp2, des2 = orb.detectAndCompute(image2,rectimg)
    image3=image2.copy()
    cv2.drawKeypoints(image3,kp2,image3)
    #cv2.imshow('image3',image3)
    #cv2.waitKey(0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    #bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    #matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)
    good = [m for (m,n) in matches if m.distance < 0.6*n.distance]

    print len(good)

    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h= image1.shape[0]
        w=image1.shape[1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        #image2 = cv2.polylines(image2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        print(M)

        img_pts1=np.dot(M,img_pts0)
        print(img_pts1)
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    des1=des2
    kp1=kp2
    img_pts0=img_pts1

    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #         singlePointColor = None,
    #         matchesMask = matchesMask, # draw only inliers
    #         flags = 2)
    # image4 = cv2.drawMatches(image1,kp1,image2,kp2,good,None,**draw_params)
    #cv2.imshow('image4',image4)

    #cv2.waitKey(0)



    image_mickey=cv2.imread('/Users/zhaopenggu/Projects/pyopencv/mickey.jpg')
    #cv2.imshow('image_mickey',image_mickey)

    img_pts3=np.zeros((3,4),dtype=np.float32)
    print(image_mickey.shape)
    h3=image_mickey.shape[0]
    w3=image_mickey.shape[1]
    img_pts3[0,0]=0
    img_pts3[1,0]=0
    img_pts3[2,0]=1
    img_pts3[0,1]=w3-1
    img_pts3[1,1]=0
    img_pts3[2,1]=1
    img_pts3[0,2]=w3-1
    img_pts3[1,2]=h3-1
    img_pts3[2,2]=1
    img_pts3[0,3]=0
    img_pts3[1,3]=h3-1
    img_pts3[2,3]=1
    M, mask = cv2.findHomography(np.transpose(img_pts3), np.transpose(img_pts1), cv2.RANSAC,5.0)

    image_mickey_1=cv2.warpPerspective(image_mickey,M,(image1.shape[1],image1.shape[0]))

    image_mickey_1_gray=cv2.cvtColor(image_mickey_1,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(image_mickey_1_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv=cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(image2,image2,mask = mask_inv)
    img2_fg = cv2.bitwise_and(image_mickey_1,image_mickey_1,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    cv2.imshow('res',dst)

    cv2.imwrite(fn_img_1,dst)
    idx1+=1
    #cv2.imshow('mask',mask)
    #cv2.imshow('image_mickey_1',image_mickey_1)
    cv2.waitKey(1)