import cv2 #step1
#STEP 1: Homomorphic Filter 
#import image
img = cv2.imread('C:\\Users\\Camila\\Pictures\\Capstone\\lesion_test.jpg',-1) # 0 = grayscale
imgg = cv2.imread('C:\\Users\\Camila\\Pictures\\Capstone\\lesion_test.jpg',0) # 0 = grayscale

img1 = np.float32(img)
img1 = img1/255
rows,cols,dim=img1.shape

rh, rl, cutoff = 2.5,0.5,32

imgYCrCb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
y,cr,cb = cv2.split(imgYCrCb)

y_log = np.log(y+0.01)

y_fft = np.fft.fft2(y_log)

y_fft_shift = np.fft.fftshift(y_fft)


DX = cols/cutoff
G = np.ones((rows,cols))
for i in range(rows):
    for j in range(cols):
        G[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl

result_filter = G * y_fft_shift

result_interm = np.real(np.fft.ifft2(np.fft.ifftshift(result_filter)))

result = np.exp(result_interm)
cv2.imshow('image1',imgg)
cv2.namedWindow('image1',WINDOW_NORMAL)
cv2.resizeWindow('image1',600,600)
cv2.imshow('image2',result)
cv2.waitKey(0)
cv2.waitKey() #necessary? -yes
