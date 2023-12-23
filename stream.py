import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import timedelta
import re
import pytesseract
import os
import cv2


def stream_files(path):
    for root, _dirs, files in os.walk("data/lake_nakaru_r/001/"):
        for shortname in sorted(files):
            filename = os.path.join(root, shortname)
            if not filename.endswith(".jpeg"):
                continue

            img = cv2.imread(filename)

            assert img is not None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img_rgb


WINDOW_NAME = "Preview"

progress_top_left = (44, 84)
progress_bottom_right = (55, 1006)

time_top_left = (1695, 45)
time_bottom_right = (1840, 100)


def stream_times(path, verbose=False, interactive=False):
    last_time = None

    for idx, file_contents in enumerate(stream_files(path)):
        time_slice = file_contents[
            time_top_left[1] : time_bottom_right[1],
            time_top_left[0] : time_bottom_right[0],
            :,
        ]

        if verbose:
            print(idx, "Processing")
        ocr_time = pytesseract.image_to_string(time_slice)
        ocr_time = ocr_time.strip(r"\w")

        matches = re.search(r"(\d+)\D(\d+)\D(\d+)", ocr_time)
        happy_regex = False
        if matches is not None and len(matches.groups()) >= 3:
            happy_regex = True

        if happy_regex:
            minutes, seconds, millis = matches.groups()
            current_time = timedelta(
                minutes=int(minutes), seconds=int(seconds), milliseconds=int(millis)
            )

            if last_time is None:
                if verbose:
                    print("Defer", idx)
                last_time = (idx, current_time)
                continue

            last_idx, last_duration = last_time

            dt = (current_time - last_duration) / (idx - last_idx)
            if dt.total_seconds() > 0.40 or dt.total_seconds() <= -0.001:
                if verbose:
                    print("Ignoring Large Dt", dt)

                if interactive:
                    print(idx, "pytesseract")
                    print(ocr_time)
                    print(idx, "Happy Regex", matches.groups())
                    cv2.imshow(WINDOW_NAME, time_slice)
                    print("Waiting for Key")
                    cv2.waitKey(0)

                continue

            for yield_idx in range(last_time[0], idx):
                if verbose:
                    print("Share", yield_idx, dt.total_seconds())
                yield yield_idx, last_duration + (dt * (yield_idx - last_idx))

            last_time = (idx, current_time)

    assert last_time is not None

    # end of loop, give the last time
    for yield_idx in range(last_time[0], idx):
        if verbose:
            print("Cleanup", idx)
        yield yield_idx, last_duration + (dt * (yield_idx - last_idx))

    if interactive:
        print("Last Frame")
        print(idx, "pytesseract")
        print(ocr_time)
        print(idx, "Happy Regex", matches.groups())
        cv2.imshow(WINDOW_NAME, time_slice)
        print("Waiting for Key")
        cv2.waitKey(0)


def plot(indexes, elapsed_time):
    elapsed_time = list(map(lambda x: x.total_seconds(), elapsed_time))
    start_time_index = len(elapsed_time) - list(reversed(elapsed_time)).index(0.0) - 1

    print("Plotting")

    travel = np.array(indexes[start_time_index:])
    elapsed_time = np.array(elapsed_time[start_time_index:])

    fig, axs = plt.subplots(figsize=(1920.0 / 300, 1080.0 / 300), ncols=1, nrows=3)

    for i in range(3):
        axs[i].set_xlabel("Travel [m] (incorrect for now)")

    top = axs[0]
    speed = 60 + 5 * np.sin(travel / 5.0)

    top.set_ylabel("Speed (mph)")
    top.set_ylim(bottom=0, top=1.1 * np.max(speed))
    top.plot(travel, speed)

    mid = axs[1]

    mid.set_ylabel("Elapsed Time (sec)")
    mid.set_ylim(bottom=0, top=1.1 * np.max(elapsed_time))
    mid.plot(travel, elapsed_time)

    bottom = axs[2]

    rpm = 4 + 2 * np.sin(travel / 2.0)

    bottom.set_ylabel("Rpm (rev / min)")
    bottom.set_ylim(bottom=0, top=1.1 * np.max(rpm))
    bottom.plot(travel, rpm)

    plt.tight_layout()
    plt.show()


def main():
    indexes = []
    elapsed_time = []

    for idx, delta_t in tqdm(stream_times("data/lake_nakaru_r/001/")):
        indexes.append(idx)
        elapsed_time.append(delta_t)

        if idx > 100 or delta_t.total_seconds() > 60.0:
            break

    plot(indexes=indexes, elapsed_time=elapsed_time)


if __name__ == "__main__":
    main()
