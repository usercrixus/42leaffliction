from plantcv import plantcv as pcv
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.morphology import remove_small_objects


def transformations(img, mask_option=False):
    # Get gaussian blur
    blur = cv2.GaussianBlur(img, (7, 7), 0)

    # Binary mask
    mask = pcv.threshold.binary(img, threshold=100, object_type='light')

    # Get ROI
    height, width = img.shape[:2]
    x, y, w, h = 0, 0, width, height
    rect_roi = pcv.roi.rectangle(img, x, y, w, h)

    a_gray = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    bin_mask = pcv.threshold.otsu(gray_img=a_gray, object_type="dark")
    cleaned_mask = remove_small_objects(
        bin_mask.astype(bool),
        max_size=49,
    ).astype(np.uint8) * 255
    filtered_mask = pcv.roi.filter(
        mask=cleaned_mask,
        roi=rect_roi,
        roi_type='partial',
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    green_layer = np.zeros_like(img)
    green_layer[:, :, 1] = filtered_mask
    output = gray_bgr.copy()
    output[filtered_mask > 0] = green_layer[filtered_mask > 0]
    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        color=(255, 0, 0),
        thickness=3,
    )

    shape_img = pcv.analyze.size(img=img, labeled_mask=filtered_mask)

    # Get pseudolandmarks
    landmarks_img = img.copy()
    landmarks_points = []

    contours, _ = cv2.findContours(
        filtered_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if contours:
        cnt = max(contours, key=cv2.contourArea)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            landmarks_points.append(("centre", (cx, cy)))
            cv2.circle(landmarks_img, (cx, cy), 6, (0, 0, 255), -1)

        top = tuple(cnt[cnt[:, :, 1].argmin()][0])
        bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])
        left = tuple(cnt[cnt[:, :, 0].argmin()][0])
        right = tuple(cnt[cnt[:, :, 0].argmax()][0])
        points = [
            ("haut", top),
            ("bas", bottom),
            ("gauche", left),
            ("droite", right),
        ]
        for name, pt in points:
            landmarks_points.append((name, pt))
            cv2.circle(landmarks_img, pt, 6, (255, 0, 0), -1)

        hull = cv2.convexHull(cnt, returnPoints=True)
        step = max(1, len(hull)//60)
        for i in range(0, len(hull), step):
            p = tuple(hull[i][0])
            landmarks_points.append((f"hull_{i}", p))
            cv2.circle(landmarks_img, p, 4, (0, 255, 255), -1)

    # Get color histogram
    hist_h, hist_w = 300, 512
    hist_img = np.ones((hist_h, hist_w, 3), dtype=np.uint8) * 255
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    colors = ('r', 'g', 'b')
    hist_data = {}

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        hist = hist / total_pixels
        hist_data[col] = hist
        cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
        for j in range(1, 256):
            color = (
                (255, 0, 0) if col == 'r'
                else (0, 255, 0) if col == 'g'
                else (0, 0, 255)
            )
            cv2.line(
                hist_img,
                (
                    int((j - 1) * hist_w / 256),
                    hist_h - int(hist[j - 1].item()),
                ),
                (int(j * hist_w / 256), hist_h - int(hist[j].item())),
                color,
                2,
            )

    return [
        ("original", img),
        ("blur", blur),
        ("mask", mask),
        ("roi", output),
        ("shape_analysis", shape_img),
        ("pseudo_landmarks", landmarks_img),
        ("landmarks_coords", landmarks_points),
        ("color_histogram_data", hist_data)
    ]


def process_image(path, dst_dir=None, mask_option=False):
    original_path = path
    img, path, filename = pcv.readimage(path)
    if img is None:
        print(f"Erreur : impossible de lire {path}")
        return

    transformed = transformations(img, mask_option)

    for name, t_img in transformed:
        if name == "landmarks_coords":
            continue

        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            if name == "color_histogram_data":
                plt.figure(figsize=(8, 4))
                colors = {'r': 'red', 'g': 'green', 'b': 'blue'}
                for col, hist in t_img.items():
                    plt.plot(hist, label=col.upper(), color=colors[col])
                plt.xlabel("Pixels intensity")
                plt.ylabel("Proportion of pixels (%)")
                plt.title("Color histogram")
                plt.legend()
                plt.grid(True)
                save_path = os.path.join(dst_dir, f"{base_name}_{name}.JPG")
                plt.savefig(save_path)
                plt.close()
                continue
            save_path = os.path.join(dst_dir, f"{base_name}_{name}.JPG")
            cv2.imwrite(save_path, t_img)

        else:
            # mode affichage interactif
            if name == "color_histogram_data":
                plt.figure(figsize=(8, 4))
                colors = {'r': 'red', 'g': 'green', 'b': 'blue'}
                for col, hist in t_img.items():
                    plt.plot(hist, label=col.upper(), color=colors[col])
                plt.xlabel("Pixels intensity")
                plt.ylabel("Proportion of pixels (%)")
                plt.title("Color histogram")
                plt.legend()
                plt.grid(True)
                plt.show()
            else:
                plt.figure(figsize=(8, 4))
                plt.imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
                plt.title(name)
                plt.axis('off')
                plt.show()


def main():
    parser = argparse.ArgumentParser(description="Leaf image transformations")
    parser.add_argument("-src", required=True, help="Source image or folder")
    parser.add_argument("-dst", help="Destination folder to save results")
    parser.add_argument(
        "-mask",
        action="store_true",
        help="Display the colored mask",
    )
    args = parser.parse_args()

    if os.path.isdir(args.src):
        # Directory
        if args.dst is None:
            print("Error: -dst is required when -src is a directory.")
            return
        pathname = Path(args.src)

        for subdir in pathname.iterdir():
            if subdir.suffix.lower() == ".jpg":
                process_image(subdir, dst_dir=args.dst, mask_option=args.mask)
            if subdir.is_dir():
                for jpg_path in subdir.rglob("*"):
                    if jpg_path.suffix.lower() == ".jpg":
                        relative_path = jpg_path.parent.relative_to(pathname)
                        dst_subdir = Path(args.dst) / relative_path
                        dst_subdir.mkdir(parents=True, exist_ok=True)

                        process_image(
                            str(jpg_path),
                            dst_dir=str(dst_subdir),
                            mask_option=args.mask,
                        )
    else:
        # File
        process_image(args.src, dst_dir=args.dst, mask_option=args.mask)


if __name__ == "__main__":
    main()
