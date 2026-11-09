import cv2
import numpy as np
from PIL import Image

def prep(src_path, out_path):
    im = cv2.imread(src_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Focus directly on head and collar
    crop = gray[5:150, 15:130]
    ch, cw = crop.shape
    
    # Square canvas with margin
    dim = max(ch, cw) + 16
    square = np.full((dim, dim), 255, dtype=np.uint8)
    square[(dim - ch) // 2 : (dim - ch) // 2 + ch, (dim - cw) // 2 : (dim - cw) // 2 + cw] = crop

    # Mild bilateral filter to remove grain while preserving facial lines
    smooth = cv2.bilateralFilter(square, 5, 25, 5)

    # Moderate CLAHE for clear eyes, nose, and jaw definition
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    prepped = clahe.apply(smooth)

    cv2.imwrite(out_path, prepped)
    print(f"Prepped {out_path} ({dim}x{dim})")

if __name__ == "__main__":
    prep("/Users/sheikhrayatkarim/OpenBot/Agents/agent-b79d76b1-b039-4f28-864e-a36e438e52e7/rayat_portrait_clean.png",
         "/Users/sheikhrayatkarim/Documents/GitHub/profile-repo/source-prepped.png")
