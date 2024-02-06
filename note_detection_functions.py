import statistics

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def show_image (title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# https://stackoverflow.com/questions/60486029/how-to-find-the-center-of-black-objects-in-an-image-with-python-opencv
def find_objects(img,org_img):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    image = img.copy()
    original_image = org_img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Applying Otsu's thresholding to create a binary image (black and white) by separating foreground and background.
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # show_image("Threshold image", thresh)

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # Extracts contours based on OpenCV version compatibility.

    objects = []

    for c in cnts:
        # Obtain bounding rectangle to get measurements
        x, y, w, h = cv2.boundingRect(c)

        # Draw the contour and center of the shape on the image
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (36, 255, 12), 1)
        objects.append(((x, y), (x+w, y+h)))

        # We are gonna use it(green boxes) for future functions, extracts each sub-image.
        # region_of_interest = image[y:y + h, x:x + w]
        # show_image("region of interest",region_of_interest)

    show_image("image with boundingBox", original_image)
    return objects

def find_object_with_template(img,org_img):
    # https://www.geeksforgeeks.org/multi-template-matching-with-opencv/
    # https://github.com/afikanyati/cadenCV/blob/master/resources/template/rest/whole_rest.jpg

    image = img.copy()
    original_image = org_img.copy()
    symbols = ['treble_clef', 'bass_clef', 'sharp', 'flat', 'rest_8', 'line','solid','time_4','empty']
    symbol_files =['templates/treble_clef2.jpg', 'templates/bass_clef.jpg', 'templates/sharp.png', 'templates/flat.png', 'templates/rest_8.png','templates/barline_1.jpg','templates/solid-note.png','templates/time4.jpg','templates/half-note-space.png']
    threshold = [0.25,0.45,0.5,0.5,0.7,0.4,0.8,0.4,0.5]


    red = (0,0,255) # color of the box
    # List for the detected symbols with coordinates & names
    symbols_info = []
    for i in range(len(symbols)):
        # Load the image and template
        template = cv2.imread(symbol_files[i])
        h, w = template.shape[0:2]

        # If there's a problem with size
        if h > image.shape[0]:
            # method 1
            difference = h - image.shape[0]
            new_h = h - difference
            scale_factor = new_h/h
            new_w = int(scale_factor*w)
            template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # method 2
            '''gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            difference = h - image.shape[0]
            temp_image = np.zeros((image.shape[0]+difference, image.shape[1]), dtype=np.uint8)
            temp_image[:, :] = 255
            temp_image[:image.shape[0], :] = gray
            image = temp_image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)'''

        # Apply template Matching
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

        # Template Matching with Multiple Objects
        threshold_ = threshold[i]
        (y_points, x_points) = np.where(res >= threshold_)

        # initialize our list of bounding boxes
        boxes = list()

        # loop over the starting (x, y) coordinates
        for (x, y) in zip(x_points, y_points):
            boxes.append((x, y, x + w, y + h))
        # to create a single bounding box
        boxes = non_max_suppression(np.array(boxes))

        # Draw the boundingBox on "image"
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(original_image, (x1,y1), (x2,y2), red, 1)

            # Tag detected object name
            text_position = (x1, y1 + h+20)
            cv2.putText(original_image, symbols[i], text_position, cv2.FONT_HERSHEY_DUPLEX, 0.5, red)

            # creating a tuple for each symbol, and appending them into symbols_info list
            symbols_info.append(tuple(( (x1,y1),(x2,y2), symbols[i])))
    show_image("sharp object with template", original_image)
    return symbols_info

# We are not using this function, it is experimental.
def remove_horizontal_lines(binaryimage,img):
    # https://docs.opencv.org/3.2.0/d1/dee/tutorial_moprh_lines_detection.html
    gray = binaryimage.copy()

    reverse = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # reverse = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, -2)

    # Remove horizontal
    extracted_img = img.copy()
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(reverse, cv2.MORPH_OPEN, horizontal_kernel)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(extracted_img, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = 255 - cv2.morphologyEx(255 - extracted_img, cv2.MORPH_CLOSE, repair_kernel)
    show_image("lines extracted", result)


'''def work_on_boxes(objects, image):
    for obj in objects:
        x1, y1 = obj[0]
        x2, y2 = obj[1]
        box = image[y1:y2, x1:x2]
        show_image("box", box)'''


def recognize_note_type(note_box, blob_type, img):
    x1 = note_box[0][0]
    x2 = note_box[1][0]
    y1 = note_box[0][1]
    y2 = note_box[1][1]
    upper_y = y1
    lower_y = y2
    if blob_type == "solid":
        upper_y -= 3
        blacks_in_row = []
        count_black = None
        is_line_under = False
        while count_black is None or count_black > 0:
            count_black = 0
            for x in range(x1, x2):
                if img[upper_y][x] == 0:
                    count_black += 1
            blacks_in_row.append(count_black)
            upper_y -= 1
            if count_black == 0 and len(blacks_in_row) < y2-y1:
                is_line_under = True
        if is_line_under:
            lower_y += 3
            blacks_in_row.clear()
            count_black = None
            while count_black is None or count_black > 0:
                count_black = 0
                for x in range(x1, x2):
                    if img[lower_y][x] == 0:
                        count_black += 1
                blacks_in_row.append(count_black)
                lower_y += 1
        line_width = statistics.mode(blacks_in_row)
        count_horizontal_lines = 0
        is_horizontal_line = False
        may_be_horizontal = 0
        still_note = True
        for blacks in blacks_in_row:
            if blacks > line_width and not still_note and not is_horizontal_line:
                if may_be_horizontal > 5:
                    is_horizontal_line = True
                    count_horizontal_lines += 1
                else:
                    may_be_horizontal += 1
            elif blacks <= line_width:
                is_horizontal_line = False
                may_be_horizontal = 0
                still_note = False
        if count_horizontal_lines == 0:
            return (x1, y1), (x2, y2), "note_4"
        elif count_horizontal_lines == 1:
            return (x1, y1), (x2, y2), "note_8"
        elif count_horizontal_lines == 2:
            return (x1, y1), (x2, y2), "note_16"
        elif count_horizontal_lines == 3:
            return (x1, y1), (x2, y2), "note_32"
    else:
        upper_y -= 3
        rows_with_blacks_counter = 0
        are_blacks = None
        while are_blacks is None or are_blacks:
            are_blacks = False
            for x in range(x1, x2):
                if img[upper_y][x] == 0:
                    are_blacks = True
                    break
            rows_with_blacks_counter += 1
            upper_y -= 1
            if rows_with_blacks_counter == y2 - y1:
                return (x1, y1), (x2, y2), "note_2"
        lower_y += 3
        rows_with_blacks_counter = 0
        are_blacks = None
        while are_blacks is None or are_blacks:
            are_blacks = False
            for x in range(x1, x2):
                if img[lower_y][x] == 0:
                    are_blacks = True
                    break
            rows_with_blacks_counter += 1
            lower_y += 1
            if rows_with_blacks_counter == y2 - y1:
                return (x1, y1), (x2, y2), "note_2"
        return (x1, y1), (x2, y2), "note_1"

