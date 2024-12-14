import cv2
import numpy as np

# Create an array of file names
images = [f'coin{i}.jpg' for i in range(1, 10)]
current_index = 0 
width,height = 500,500

def filter_colors(image):
    global width, height
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    resized_image = cv2.resize(image_hsv, (width, height))

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([40, 255, 255])
    blue_lower = np.array([95, 110, 110])
    blue_upper = np.array([255, 255, 255])

    yellow_mask = cv2.inRange(resized_image, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(resized_image, blue_lower, blue_upper)
    # yellow_mask = cv2.medianBlur(yellow_mask, 3)
    # blue_mask = cv2.medianBlur(blue_mask, 5)

    #for yellowwwwwwwwwwwwwwwwww
    kernel_erode = np.ones((9, 9), np.uint8)
    yellow_mask = cv2.erode(yellow_mask, kernel_erode, iterations=1)

    kernel_dilate = np.ones((7, 7), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel_dilate, iterations=1) 

#     #for blueeeeeeeeeeeeeeeeeeeee
    kernel_erode = np.ones((8, 8), np.uint8)
    blue_mask = cv2.erode(blue_mask, kernel_erode, iterations=1)
    # kernel_erode = np.array([
    # [0, 0, 0, 0, 1],
    # [0, 0, 0, 1, 0],
    # [0, 0, 1, 0, 0],
    # [0, 1, 0, 0, 0],
    # [1, 0, 0, 0, 0]
    # ], dtype=np.uint8)
    # blue_mask = cv2.erode(blue_mask, kernel_erode, iterations=1)
#     kernel_erode = np.array([
#     [0, 0, 1],
#     [0, 1, 0],
#     [1, 0, 0]
# ], dtype=np.uint8)
#     blue_mask = cv2.erode(blue_mask, kernel_erode, iterations=1)
#     blue_mask = cv2.erode(blue_mask, kernel_erode, iterations=1)

    # kernel_op = np.ones((1, 0), np.uint8)
    # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel_op)
    
    # kernel_cl = np.ones((3, 3), np.uint8)
    # blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_cl)

    # kernel_dilate = np.ones((5,5), np.uint8)
    # blue_mask = cv2.dilate(blue_mask, kernel_dilate, iterations=1)

    # kernel_erode = np.ones((5, 3), np.uint8)
    # blue_mask = cv2.erode(blue_mask, kernel_erode, iterations=1)

   

    contours_yellow, hierarchy_yellow = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_blue, hierarchy_blue = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    yellow = len(contours_yellow)
    blue = len(contours_blue)
    
    combined_mask = cv2.bitwise_or(yellow_mask, blue_mask)
    filtered_image = cv2.bitwise_and(resized_image, resized_image, mask=combined_mask)
    cv2.putText(filtered_image, (f'[Y {yellow}, B {blue}]'), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Show the original and filtered images
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_image)

    return [yellow, blue]

while True:
    image = cv2.imread(images[current_index])
    image = cv2.resize(image, (width, height))

    # Check if the image is loaded successfully
    if image is None:
        print(f"Could not load {images[current_index]}")
        break

    # Apply color filtering
    coin_counting = filter_colors(image)

    # Wait for a key press
    key = cv2.waitKey(0)

    if key == ord('d'):  # Press 'd' to move to the next image
        current_index = (current_index + 1) % len(images)
    elif key == ord('a'):  # Press 'a' to move to the next image
        current_index = (current_index - 1)  % len(images)
    elif key == ord('q'):  # Press 'q' to quit
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
