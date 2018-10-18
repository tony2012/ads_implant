import cv2



def mouse_calback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('lbtd:'+str(x)+','+str(y))
        #print(H_ou_global)
        pt = np.zeros((3, 1), np.float32)
        pt[0] = x
        pt[1] = y
        pt[2] = 1.0
        print(pt)
        pt_obj=np.dot(H_wu_global,pt)
        pt_obj=np.divide(pt_obj,pt_obj[2])
        print(pt_obj)
        pt_tab=np.dot(H_tw_global,pt_obj)
        print(pt_tab)


videoCapture = cv2.VideoCapture('/Users/zhaopenggu/Downloads/h07133bgsz6.mp4')  

fps = videoCapture.get(cv2.CAP_PROP_FPS)  
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
print(size)
success, frame = videoCapture.read()  

idx=0
while success :  
    cv2.imshow("Oto Video", frame) 
    cv2.waitKey(1000/int(fps))
    success, frame = videoCapture.read()
    fn_img="%05d.jpg" % idx
    #cv2.imwrite(fn_img,frame)
    idx+=1
    print(idx)  