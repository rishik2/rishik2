import cv2, glob

images = glob.glob("*.jpg")

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for image in images:
    img = cv2.imread(image)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detect.detectMultiScale(grey_img, 1.1, 3)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow("Images", img)
    cv2.waitKey(2000)

    cv2.destroyAllWindows()






