import cv2
import numpy as np

def preprocess_image(image_path, method="adaptive"):
    """
    Reads an image, converts to grayscale, applies Gaussian blur, and binarizes it.
    Supports either 'otsu' or 'adaptive' thresholding.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise removal
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarization
    if method == "otsu":
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Adaptive Gaussian Thresholding (helps with shadows/uneven lighting)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    return img, gray, binary

def correct_skew(image, binary_image):
    """
    Detects text skew and rotates the image to correct it.
    """
    coords = np.column_stack(np.where(binary_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated_bin = cv2.warpAffine(binary_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated, rotated_bin

def segment_lines(binary_image):
    """
    Segments a paragraph into individual lines using horizontal projection profiles.
    Returns a list of bounding boxes (x, y, w, h) for each line.
    """
    proj = np.sum(binary_image, axis=1)
    
    proj_blurred = np.convolve(proj, np.ones(5)/5, mode='same')
    
    threshold = np.max(proj_blurred) * 0.1
    active = proj_blurred > threshold
    
    line_boxes = []
    start = -1
    for i, val in enumerate(active):
        if val and start == -1:
            start = i
        elif not val and start != -1:
            end = i
            if end - start > 10: 
                line_boxes.append((0, start, binary_image.shape[1], end - start))
            start = -1
            
    if start != -1 and len(active) - start > 10:
        line_boxes.append((0, start, binary_image.shape[1], len(active) - start))
        
    return line_boxes

def get_word_bounding_boxes(line_img):
    """
    Simple word segmentation using findContours
    """
    # Dilate vertically to connect letters in words
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(line_img, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    word_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # Filter out noise
            word_boxes.append((x, y, w, h))
            
    # Sort boxes from left to right
    word_boxes = sorted(word_boxes, key=lambda b: b[0])
    return word_boxes

if __name__ == "__main__":
    import yaml
    import pandas as pd
    import os
    
    print("Preprocessing module ready.")
