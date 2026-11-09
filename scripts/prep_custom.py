import cv2
import numpy as np
from PIL import Image

# Read source-photo.jpg
img = cv2.imread('/Users/sheikhrayatkarim/Documents/GitHub/profile-repo/source-photo.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to high detail
h, w = gray.shape
# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# Smooth noise
smooth = cv2.bilateralFilter(enhanced, 9, 75, 75)

# Save prepped
cv2.imwrite('/Users/sheikhrayatkarim/Documents/GitHub/profile-repo/source-prepped.png', smooth)
print("Saved source-prepped.png successfully!")
