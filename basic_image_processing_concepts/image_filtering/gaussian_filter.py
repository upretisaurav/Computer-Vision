import cv2


img = cv2.imread("../images/potrait.jpeg")

blurred = cv2.GaussianBlur(img, (15, 15), 0)

cv2.imshow("Original Image", img)
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
