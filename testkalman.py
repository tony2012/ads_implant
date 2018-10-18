import cv2
import pickle
import numpy as np

H_tw = np.array([[1.0, 0.0, 0.05119999870657921], [0.0, -1.0, 0.179299995303154], [0.0, 0.0, 1.0]],dtype=np.float32)
D = np.array([ -0.080161, 0.183441, -0.000282, 0.000307, 0.066149],dtype=np.float32)
K = np.array([[ 908.893481, 0.000000, 520.704588], [0.000000, 908.649028, 379.023983], [0.000000, 0.000000, 1.000000]],dtype=np.float32)
K_inv = np.array([[ 0.001100, 0.000000, -0.572899], [0.000000, 0.001101, -0.417129], [0.000000, 0.000000, 1.000000]],dtype=np.float32)
H_wu = np.array([[ -0.001116, 0.000169, 0.959178], [0.000005, -0.001135, 0.500119], [-0.000020, 0.000216, 0.503912]],dtype=np.float32)
H_uw = np.array([[-923.844526, 165.073482, 1594.673547], [-17.320726, -737.873247, 765.288714], [-0.029513, 0.322480, 1.720598]],dtype=np.float32)

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.0001
kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.001

last_mes = current_mes = np.array((2,1),np.float32)
last_pre = current_pre = np.array((2,1),np.float32)

f=open('/Users/zhaopenggu/Projects/tracking_analysis/tracking.txt', 'r')
data = f.readlines() 


counter = 0
for line in data:
    odom = line.split()
    numbers_float = map(float, odom)
    print numbers_float[0]

    last_pre = current_pre
    last_mes = current_mes
    current_mes = np.array([[numbers_float[4]],[numbers_float[5]]])

    
    
    # print(current_mes)
    # if counter == 0:
    #     current_pre = kalman.predict()
    #     kalman.correct(current_mes)
    # else:
    #     current_pre = kalman.predict()
    #     kalman.correct(current_mes)

    
    counter+=1
        
    
