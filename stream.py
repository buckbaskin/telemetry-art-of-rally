import numpy as np
from datetime import timedelta
import re
import pytesseract
import os
import cv2


def stream_files(path):
    for root, dirs, files in os.walk("data/lake_nakaru_r/001/"):
        for shortname in sorted(files):
            filename = os.path.join(root, shortname)
            if not filename.endswith(".jpeg"):
                continue

            img = cv2.imread(filename)

            assert img is not None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img


WINDOW_NAME = "Preview"

progress_top_left = (44, 84)
progress_bottom_right = (55, 1006)

time_top_left = (1695, 45)
time_bottom_right = (1840, 100)


def stream_times(path):
    last_time = None

    for idx, file_contents in enumerate(stream_files(path)):
        time_slice = file_contents[
            time_top_left[1] : time_bottom_right[1],
            time_top_left[0] : time_bottom_right[0],
            :,
        ]

        print(idx, "Processing")
        ocr_time = pytesseract.image_to_string(time_slice)
        ocr_time = ocr_time.strip(r"\w")

        matches = re.search("(\d+)\D(\d+)\D(\d+)", ocr_time)
        happy_regex = False
        if matches is not None and len(matches.groups()) >= 3:
            happy_regex = True

        if happy_regex:
            minutes, seconds, millis = matches.groups()
            current_time = timedelta(
                minutes=int(minutes), seconds=int(seconds), milliseconds=int(millis)
            )

            if last_time is None:
                print("Defer", idx)
                last_time = (idx, current_time)
                continue

            last_idx, last_duration = last_time

            dt = (current_time - last_duration) / (idx - last_idx)
            if dt.total_seconds() > 0.40 or dt.total_seconds() <= -0.001:
                print("Ignoring Large Dt", dt)

                print(idx, "pytesseract")
                print(ocr_time)
                print(idx, "Happy Regex", matches.groups())
                cv2.imshow(WINDOW_NAME, time_slice)
                print("Waiting for Key")
                cv2.waitKey(0)
                continue

            for yield_idx in range(last_time[0], idx):
                print("Share", yield_idx, dt.total_seconds())
                yield yield_idx, last_duration + (dt * (yield_idx - last_idx))

            last_time = (idx, current_time)

    assert last_time is not None

    # end of loop, give the last time
    for yield_idx in range(last_time[0], idx):
        print("Cleanup", idx)
        yield yield_idx, last_duration + (dt * (yield_idx - last_idx))

    print("Last Frame")
    print(idx, "pytesseract")
    print(ocr_time)
    print(idx, "Happy Regex", matches.groups())
    cv2.imshow(WINDOW_NAME, time_slice)
    print("Waiting for Key")
    cv2.waitKey(0)
    continue


for idx, delta_t in stream_times("data/lake_nakaru_r/001/"):
    print(idx, delta_t, delta_t.total_seconds())
