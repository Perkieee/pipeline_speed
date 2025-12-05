import cv2

import numpy as np

import time

import os

from picamera2 import Picamera2



MODE = "accuracy"          # "accuracy" or "throughput"

RUN_TIME_SECONDS = 180     # used only in throughput mode

CROP_SAVE_DIR = "/home/sir/Kevin_Walter/tests/edge_real_time/crops"    # where to save crops in accuracy mode



# Edge and contour parameters

MIN_CONTOUR_AREA_RATIO = 0.01  # ~1% of the frame area

MIN_CONTOUR_AREA_FLOOR = 500   # never go below this area

MARGIN_PIXELS = 10             # padding around detected object for crops



def ensure_dir(path):

    if not os.path.exists(path):

        os.makedirs(path)



def main():

    picam2 = Picamera2()



    camera_config = picam2.create_video_configuration(

        main={"size": (640, 480), "format": "BGR888"}

    )

    picam2.configure(camera_config)

    picam2.start()



    time.sleep(1)

    print(f"IMX500 started in mode: {MODE}")

    print("Press q to quit early.")



    if MODE == "accuracy":

        ensure_dir(CROP_SAVE_DIR)



    total_frames = 0

    total_crops = 0



    start_time = time.time()



    while True:

        # Stop automatically after RUN_TIME_SECONDS in throughput mode

        if MODE == "throughput":

            elapsed = time.time() - start_time

            if elapsed >= RUN_TIME_SECONDS:

                print("Run time reached, exiting loop")

                break



        frame_rgb = picam2.capture_array()

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)



        # Color-based mask to find the banana (assumes yellow object on grey desk)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([18, 30, 120], dtype=np.uint8)

        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)



        # Clean the mask to get a solid blob

        kernel = np.ones((5, 5), np.uint8)

        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)



        # Find contours on the cleaned mask

        contours, _ = cv2.findContours(

            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE

        )



        frame_crops = 0

        h_img, w_img = frame_bgr.shape[:2]

        image_area = h_img * w_img

        min_contour_area = max(

            int(image_area * MIN_CONTOUR_AREA_RATIO), MIN_CONTOUR_AREA_FLOOR

        )



        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < min_contour_area:

                continue



            x, y, w, h = cv2.boundingRect(cnt)



            # Add a margin and clip to image bounds

            x = max(0, x - MARGIN_PIXELS)

            y = max(0, y - MARGIN_PIXELS)

            x2 = min(w_img, x + w + 2 * MARGIN_PIXELS)

            y2 = min(h_img, y + h + 2 * MARGIN_PIXELS)



            crop = frame_bgr[y:y2, x:x2]

            if crop.size == 0:

                continue



            frame_crops += 1



            if MODE == "accuracy":

                timestamp = int(time.time() * 1000)

                filename = f"crop_{timestamp}_{total_frames}_{frame_crops}.jpg"

                save_path = os.path.join(CROP_SAVE_DIR, filename)

                cv2.imwrite(save_path, crop)



        total_frames += 1

        total_crops += frame_crops



        # For visual sanity check, draw boxes on the original frame

        overlay = frame_bgr.copy()

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < min_contour_area:

                continue

            x, y, w, h = cv2.boundingRect(cnt)

            x = max(0, x - MARGIN_PIXELS)

            y = max(0, y - MARGIN_PIXELS)

            x2 = min(w_img, x + w + 2 * MARGIN_PIXELS)

            y2 = min(h_img, y + h + 2 * MARGIN_PIXELS)

            cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 255), 1)



        cv2.imshow("IMX500 overlay", overlay)

        cv2.imshow("Mask", mask_clean)



        # Press q to quit manually

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break



    end_time = time.time()

    elapsed = end_time - start_time



    picam2.stop()

    cv2.destroyAllWindows()



    print("Summary")

    print(f"Mode: {MODE}")

    print(f"Total frames processed: {total_frames}")

    print(f"Total crops: {total_crops}")

    print(f"Total time: {elapsed:.2f} seconds")



    if elapsed > 0:

        fps = total_frames / elapsed

        print(f"Pipeline throughput: {fps:.2f} frames per second")





if __name__ == "__main__":

    main()

