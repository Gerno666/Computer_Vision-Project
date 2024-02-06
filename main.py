from general_functions import *
from staff_detection import estimate_line_distance, staff_detection
from note_detection_functions import *
from interpretation import create_sounds_sequence, check_is_natural
from music import play_note
import pygame

# Step 1. Image processing

#img = cv2.imread("simple_test.png")
img = cv2.imread("test_images/semisimple.png") #For detecting "bass clef"

bin_img = binarize(img, 80)

# show binary image
cv2.imshow("Bin", bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

height = bin_img.shape[0]
width = bin_img.shape[1]

# Step 2. Staff detection

estimated_distance_between_lines = estimate_line_distance(bin_img)
staffs = staff_detection(bin_img, estimated_distance_between_lines)

if len(staffs) == 1:
    print("1 staff found")
else:
    print(len(staffs), " staffs found")

#for staff in staffs:
    #show(staff, "staff found", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# Step 3. Split image into subimages containing one staff

subimages = []
previous_border = 0
subimages_upper_borders = []
for i in range(len(staffs)-1):
    subimage_border = None
    if staffs[i][4][1] >= staffs[i][4][3]:
        subimage_border = staffs[i][4][1] + int((staffs[i+1][0][1] - staffs[i][4][1])/2)
    else:
        subimage_border = staffs[i][4][3] + int((staffs[i + 1][0][3] - staffs[i][4][3]) / 2)
    upper_border = previous_border - abs(staffs[i][0][1] - staffs[i][0][3])
    if upper_border < 0:
        upper_border = 0
    subimages.append(img[upper_border:subimage_border, :])
    subimages_upper_borders.append(upper_border)
    previous_border = subimage_border
subimages.append(img[previous_border:, :])
subimages_upper_borders.append(previous_border)

# for image in subimages:
#     cv2.imshow("sub", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

pygame.init()

for index in range(len(subimages)):

    # Step 4. Detecting and recognizing objects - symbols

    binImg=binarize2(subimages[index],150)

    # Inverse binary image
    binImg = cv2.bitwise_not(binImg)

    # Create the images that will use to extract the horizontal and vertical lines
    verticalImg = binImg.copy()

    # Specify size on horizontal axis for structuring element (kernel)
    h, w = verticalImg.shape  # height, width
    horizontalSize = round(w / 15)

    # Specify size on vertical axis for structuring element (kernel)
    verticalSize = round(h / 50)  #for Complex_test.png => 5 pixel is kinda better.

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))

    # Apply morphology operations
    verticalImg = cv2.erode(verticalImg, verticalStructure)
    verticalImg = cv2.dilate(verticalImg, verticalStructure)

    # Show extracted vertical lines
    # show_image("extracted vertical image", verticalImg)

    # Inverse vertical image
    verticalImg = cv2.bitwise_not(verticalImg)
    # show_image("Inverse vertical image", verticalImg)

    #we transferred the Black&white image into RGB
    verticalImgRGB = cv2.cvtColor(verticalImg, cv2.COLOR_GRAY2RGB)

    #https://stackoverflow.com/questions/60486029/how-to-find-the-center-of-black-objects-in-an-image-with-python-opencv
    objects = find_objects(verticalImgRGB,subimages[index])


    # Template Matching
    symbols=find_object_with_template(verticalImgRGB,subimages[index])
    #print(symbols)

    # notes = work_on_boxes(objects, verticalImgRGB)

    for i in range(len(symbols)):
        if symbols[i][2] == "solid":
            symbol = recognize_note_type((symbols[i][0], symbols[i][1]), "solid",
                                         cv2.cvtColor(verticalImgRGB, cv2.COLOR_BGR2GRAY))
            symbols[i] = symbol
        elif symbols[i][2] == "empty":
            symbol = recognize_note_type((symbols[i][0], symbols[i][1]), "empty",
                                         cv2.cvtColor(verticalImgRGB, cv2.COLOR_BGR2GRAY))
            symbols[i] = symbol
        elif symbols[i][2] == "sharp":
            symbol = check_is_natural((symbols[i][0], symbols[i][1]), cv2.cvtColor(verticalImgRGB, cv2.COLOR_BGR2GRAY))
            symbols[i] = symbol
    print(symbols)

    # Step 5. Interpretation of symbols

    #index = subimages.index(image)
    staff = staffs[index]
    for line in staff:
        line[1] -= subimages_upper_borders[index]
        line[3] -= subimages_upper_borders[index]
    sound_sequence = create_sounds_sequence(staff, symbols)

    # Step 6. Playing sounds

    if sound_sequence is None:
        continue
    else:
        for note in sound_sequence:
            noteName = 'r'
            if note[1] != 'r':
                octave = note[2] + 4
                noteName = note[1]+str(octave)
            else:
                noteName = 'r'
            play_note(noteName, 3.0/note[0])

pygame.quit()
