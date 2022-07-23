import cv2
from pygments import highlight

img = cv2.imread('imagem.png')

#Cor
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#Tudo que estiver dentro do range indicado
filtered = cv2.inRange(hsv, (160,150,150), (179,255,255))

#Contorno
contours,hierarchy = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
    if len(approx) == 3:
        cv2.drawContours(img, (contour), -1, (0,0,0), 4)

cv2.imshow('imagem', filtered)#mostra a imagem com os filtros de cores aplicados, no caso vermelho

#print(img[0][0][0])
#print(img[0][0][1])
#print(img[0][0][2])

#img[:,:,0] = 0

#cv2.imshow('imagem', img)#mostra a imagem com o contorno
cv2.waitKey(0)