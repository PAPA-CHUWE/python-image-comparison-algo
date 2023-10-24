import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

pathImg = "AlgoPicToPic/Assets/image.jpg"  # Specify the path and filename for saving the image

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error opening camera")
    exit()

while True:
    ret, frame = cam.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the resulting frame
    cv2.imshow('frame', img)
    
    if cv2.waitKey(1) == ord('q'):
        cv2.imwrite(pathImg, frame)  # Save the captured frame as an image
        break

cam.release()
cv2.destroyAllWindows()

def compare_images(pathImg, img):
    img1 = cv2.imread(pathImg)
    img2 = img

    # Perform image comparison using SSIM
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    similarity_score = ssim(gray_img1, gray_img2)
    similarity_percentage = similarity_score * 100

    # Display the two images side by side with the similarity percentage
    final_frame = cv2.hconcat([img1, img2])
    cv2.putText(final_frame, f"Similarity: {similarity_percentage:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Comparison', final_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the compare_images function
compare_images(pathImg, img)