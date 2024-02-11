import sys
import cv2
import numpy as np
import utils

MIN_SQ_AREA = 100.0 ** 2

YELLOW_AREA    = 3400.0
RED_AREA       = 3600.0
PURPLE_AREA    = 1700.0
MIN_FRUIT_AREA = 500.0

TOO_LARGE_FACTOR = 1.6


def double_centroids(contour):
  major_centroid = np.mean(contour, axis=0)

  distances = np.linalg.norm(
    contour.reshape(-1, 2) - major_centroid, axis=1)
  
  sorted_indices = np.argsort(distances)
  smallest_two_indices = sorted_indices[:2]

  i1 = min(smallest_two_indices)
  i2 = max(smallest_two_indices)

  contour0_0 = contour[:i1]
  contour1   = contour[i1:i2]
  contour0_1 = contour[i2:]

  contour0 = np.concatenate((contour0_0, contour0_1), axis=0)

  centroid0 = np.mean(contour0, axis=0)
  centroid1 = np.mean(contour1, axis=0)

  return (centroid0, centroid1)

  


def get_coords(raw_image, given_color):
  image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

  # white mask
  lower_white = np.array([0, 0, 200]); upper_white = np.array([179, 30, 255])
  white_mask = cv2.inRange(image, lower_white, upper_white)

  # green mask
  lower_green = np.array([40, 40, 40]); upper_green = np.array([80, 255, 255])
  green_mask = cv2.inRange(image, lower_green, upper_green)

  combined_mask = cv2.bitwise_or(white_mask, green_mask)

  """ removing any square outside central square """
  height, width = image.shape[:2]
  rect_width = int(width * 0.6)
  rect_height = int(height * 0.75)

  x1 = int((width - rect_width) / 2)
  y1 = int((height - rect_height) / 2)
  x2 = x1 + rect_width
  y2 = y1 + rect_height

  central_mask = np.zeros_like(image, dtype=np.uint8)
  cv2.rectangle(central_mask, (x1, y1), (x2, y2), (255, 255, 255), -1)  

  tmp = img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
  combined_mask = cv2.bitwise_and(tmp, central_mask)

  """ getting region of interest """
  gray = 255 - combined_mask
  blur = cv2.GaussianBlur(gray, (3,3), 0)
  gray_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
  thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 3)

  # Morph open
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

  # Find contours
  cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  # Initialize variables for the largest rectangle
  max_area = 0
  max_rect = None

  # Iterate through all contours
  for contour in cnts:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    area = cv2.contourArea(contour)

    if area > max_area:
      max_area = area
      max_rect = box

  if max_area < MIN_SQ_AREA:
    print("No close plant bed found")
    return []


  # Draw the largest rectangle
  interested_region = combined_mask.copy()

  if len(cnts) > 0:
    cv2.drawContours(interested_region, [max_rect], 0, (36, 255, 12), 2)
    cv2.fillPoly(interested_region, [max_rect], (255, 255, 255))
  else:
    print("No plant bed found")
    return []

  """ cropping out region of interest """
  mask = np.zeros_like(raw_image)
  cv2.fillPoly(mask, [max_rect], color=(255, 255, 255))

  cropped_img = cv2.bitwise_and(raw_image, mask)

  """ detecting contours in ROI """
  basic_contour = cropped_img.copy()
  hsv = cv2.cvtColor(basic_contour, cv2.COLOR_BGR2HSV)

  lower_red = np.array([0, 100, 100]); upper_red = np.array([10, 255, 255])
  lower_yellow = np.array([20, 100, 100]); upper_yellow = np.array([30, 255, 255])
  lower_purple = np.array([130, 50, 50]); upper_purple = np.array([160, 255, 255])

  good_area = 0
  combined_mask_color = None
  if given_color == "Tomato":
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    combined_mask_color = red_mask
    good_area = RED_AREA
  elif given_color == "Pepper":
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    combined_mask_color = yellow_mask
    good_area = YELLOW_AREA
  elif given_color == "Eggplant":
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    combined_mask_color = purple_mask
    good_area = PURPLE_AREA

  contours, _ = cv2.findContours(combined_mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  cv2.drawContours(basic_contour, contours, -1, (0, 255, 0), 2)

  """ check area of each contour """
  too_large_area = good_area * TOO_LARGE_FACTOR

  centers = []

  for idx, contour in enumerate(contours):
    contour_area = cv2.contourArea(contour)

    if contour_area < MIN_FRUIT_AREA:
      continue
    elif contour_area > too_large_area:
      centroid0, centroid1 = utils.double_centroids(contour)
      centers.append((idx, centroid0[0]))
      centers.append((idx, centroid1[0]))
    else:
      centroid = np.mean(contour, axis=0)
      centers.append((idx, centroid[0]))
          
  # computing vector from center of bed
  bed_center = np.mean(max_rect, axis=0)
  outputs = []
  for point in centers:
    outputs.append(point[1] - bed_center)

  return outputs

if __name__ == "__main__":
  img_file = sys.argv[1]
  raw_image = cv2.imread(f"./data/{img_file}.png")

  print(get_coords(raw_image, "Pepper"))