import cv2


def binarize(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh, bin = cv2.threshold(gray, threshold, 255, 0)
    return bin


def binarize2(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, bin = cv2.threshold(gray, threshold, 255, 0)
    return bin

# shows detected lines on image - WARNING!!! IT WORKS ONLY IF img ARGUMENT IS IMAGE IN GRAY SCALE!
def show(lines, title, img):
    img_to_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(img_to_show, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow(title, img_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image(title,image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
