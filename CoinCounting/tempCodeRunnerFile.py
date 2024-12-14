   kernel_dilate = np.ones((15,15), np.uint8)
    blue_mask = cv2.dilate(blue_mask, kernel_dilate, iterations=1)