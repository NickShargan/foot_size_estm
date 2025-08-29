import argparse
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import SAM


# Mondopoint (cm) reference to EU/US/UK - adults (approx.)
# MEN
MEN_TABLE = {
    24.5: {"EU": 39,   "US": 6.5, "UK": 6},
    25.0: {"EU": 40,   "US": 7.0, "UK": 6.5},
    25.5: {"EU": 40.5, "US": 7.5, "UK": 7},
    26.0: {"EU": 41,   "US": 8.0, "UK": 7.5},
    26.5: {"EU": 42,   "US": 8.5, "UK": 8},
    27.0: {"EU": 42.5, "US": 9.0, "UK": 8.5},
    27.5: {"EU": 43,   "US": 9.5, "UK": 9},
    28.0: {"EU": 44,   "US": 10.0, "UK": 9.5},
    28.5: {"EU": 44.5, "US": 10.5, "UK": 10},
    29.0: {"EU": 45,   "US": 11.0, "UK": 10.5},
    29.5: {"EU": 46,   "US": 11.5, "UK": 11},
    30.0: {"EU": 46.5, "US": 12.0, "UK": 11.5},
}

# WOMEN
WOMEN_TABLE = {
    22.0: {"EU": 35,   "US": 5.0, "UK": 2.5},
    22.5: {"EU": 35.5, "US": 5.5, "UK": 3.0},
    23.0: {"EU": 36,   "US": 6.0, "UK": 3.5},
    23.5: {"EU": 37,   "US": 6.5, "UK": 4.0},
    24.0: {"EU": 37.5, "US": 7.0, "UK": 4.5},
    24.5: {"EU": 38,   "US": 7.5, "UK": 5.0},
    25.0: {"EU": 39,   "US": 8.0, "UK": 5.5},
    25.5: {"EU": 39.5, "US": 8.5, "UK": 6.0},
    26.0: {"EU": 40,   "US": 9.0, "UK": 6.5},
    26.5: {"EU": 41,   "US": 9.5, "UK": 7.0},
    27.0: {"EU": 42,   "US": 10.0, "UK": 7.5},
}

# Width classification tables (approximate, in millimeters)
# Based on US width letters. Manufacturers may differ slightly.
MEN_WIDTH_TABLE = {
    "2A": (0, 84),     # Extra Narrow
    "B": (85, 94),     # Narrow
    "D": (95, 104),    # Standard
    "2E": (105, 114),  # Wide
    "4E": (115, 999),  # Extra Wide
}

WOMEN_WIDTH_TABLE = {
    "4A": (0, 69),     # Slim
    "2A": (70, 79),    # Narrow
    "B": (80, 89),     # Standard
    "D": (90, 99),     # Wide
    "2E": (100, 999),  # Extra Wide
}


def pick_size_mm_for_top_edge(corners_img, base_size_mm=(215.9, 279.4)):
    """
    corners_img: [TL, TR, BR, BL] in pixels.
    base_size_mm: (short, long) side, e.g. (215.9, 279.4) for Letter,
                  (210, 297) for A4.
    Returns (W, H) so that the TOP edge (TL->TR) maps to the LONGER mm side
    if the top edge is longer in the image; otherwise to the shorter side.
    """
    corners = np.asarray(corners_img, float)
    TL, TR, BR, BL = corners
    top_len_px = np.linalg.norm(TR - TL)
    side_len_px = np.linalg.norm(BR - TR)

    short_mm, long_mm = sorted(base_size_mm)
    if top_len_px >= side_len_px:
        # top edge looks longer in the image -> map to long side
        return (long_mm, short_mm)
    else:
        # top edge looks shorter in the image -> map to short side
        return (short_mm, long_mm)


def order_corners(pts):
    """Return corners as (top-left, top-right, bottom-right, bottom-left)."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_ref_obj_corners(paper_mask):
    """return coordinates of segmentation mask of paper (reference object)
    to use them for homography."""

    top_left_p = [-1, -1]
    top_right_p = [-1, -1]
    bottom_left_p = [-1, -1]
    bottom_right_p = [-1, -1]

    heigh, width = paper_mask.shape

    for yy in range(heigh):
        for xx in range(width):
            if not paper_mask[yy, xx]:
                continue

            if top_left_p[0] == -1:
                top_left_p[0] = xx
                top_left_p[1] = yy

                top_right_p[0] = xx
                top_right_p[1] = yy

                bottom_left_p[0] = xx
                bottom_left_p[1] = yy

                bottom_right_p[0] = xx
                bottom_right_p[1] = yy

            if xx + yy < top_left_p[0] + top_left_p[1]:
                top_left_p[0] = xx
                top_left_p[1] = yy

            if xx + (heigh - yy) > top_right_p[0] + (heigh - top_right_p[1]):
                top_right_p[0] = xx
                top_right_p[1] = yy

            if width - xx + yy > (width - bottom_left_p[0]) + bottom_left_p[1]:
                bottom_left_p[0] = xx
                bottom_left_p[1] = yy

            if xx + yy > bottom_right_p[0] + bottom_right_p[1]:
                bottom_right_p[0] = xx
                bottom_right_p[1] = yy

    points = np.array([top_left_p, top_right_p, bottom_right_p, bottom_left_p])

    return points


def get_foot_points(foot_mask):
    """Returns coordinates of 3 points of foot: 1) the longest toe;
    2) the most right point; 3) the most left point."""

    toe_p = [-1, -1]
    fright_p = [-1, -1]
    fleft_p = [-1, -1]

    heigh, width = foot_mask.shape

    for yy in range(heigh):
        for xx in range(width):
            if not foot_mask[yy, xx]:
                continue

            if toe_p[0] == -1:
                toe_p[0] = xx
                toe_p[1] = yy

                fright_p[0] = xx
                fright_p[1] = yy

                fleft_p[0] = xx
                fleft_p[1] = yy

            if yy < toe_p[1]:
                toe_p[0] = xx
                toe_p[1] = yy

            if xx > fright_p[0]:
                fright_p[0] = xx
                fright_p[1] = yy

            if xx < fleft_p[0]:
                fleft_p[0] = xx
                fleft_p[1] = yy

    points_f = np.array([toe_p, fright_p, fleft_p])

    return points_f


def project_points_to_top_edge(
        corners_img,  # [TL, TR, BR, BL] in image pixels, shape (4,2)
        pts_img,      # points to project, shape (N,2)
        size_mm=(215.9, 279.4),  # Letter by default (W,H) in mm; (210,297) for A4
        camera_matrix=None,      # 3x3 if you have calibration
        dist_coeffs=None         # (k1,k2,p1,p2, ...), if you have calibration
    ):

    corners_img = np.asarray(corners_img, dtype=np.float32)
    pts_img = np.asarray(pts_img, dtype=np.float32)

    # 0) Optional: undistort the image points and the 4 corners
    if camera_matrix is not None and dist_coeffs is not None:
        def undistort(pts):
            return cv2.undistortPoints(
                pts.reshape(-1, 1, 2), camera_matrix, dist_coeffs,
                P=camera_matrix
            ).reshape(-1, 2).astype(np.float32)

        corners_img = undistort(corners_img)
        pts_img = undistort(pts_img)

    # 1) Build homography from image->paper coordinates
    W, H = size_mm
    # TL,TR,BR,BL
    dst_paper = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    H_img2paper = cv2.getPerspectiveTransform(corners_img, dst_paper)
    H_paper2img = np.linalg.inv(H_img2paper)

    # 2) Map points into the paper plane
    pts_paper = cv2.perspectiveTransform(pts_img.reshape(-1, 1, 2),
                                         H_img2paper).reshape(-1, 2)

    # 3) Orthogonal projection onto the top edge y=0, clamped to the segment
    # [x=0..W]
    x = np.clip(pts_paper[:, 0], 0, W)
    projs_paper = np.column_stack([x, np.zeros_like(x)]).astype(np.float32)

    # Distances to the top edge in *millimeters* on the paper (signed by
    # convention)
    # use np.abs(...) if you want always-positive
    dist_to_top_mm = pts_paper[:, 1].copy()

    # 4) Map the projected points back to the original image
    projs_img = cv2.perspectiveTransform(projs_paper.reshape(-1, 1, 2),
                                         H_paper2img).reshape(-1, 2)

    return projs_img, projs_paper, dist_to_top_mm


def maks_vis(img, corners_img, points_img, projs_img,
             labels=None, dist_mm=None):
    """
    img:          HxW[x3] array shown with plt.imshow(...)
    corners_img:  4x2 array/list in order [TL, TR, BR, BL]  (used to draw top
                  edge)
    points_img:   Nx2 array of original points (e.g., toe, most_left,
                  most_right)
    projs_img:    Nx2 array of projected points on the top edge (output from
                  your function)
    labels:       list of N strings to annotate original points
    dist_mm:      optional length-N array of distances in mm (if you computed
                  them)
    """
    img = np.asarray(img)
    points_img = np.asarray(points_img, dtype=float)
    projs_img = np.asarray(projs_img, dtype=float)
    corners_img = np.asarray(corners_img, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)

    # Draw the top edge (TL -> TR) for context
    TL, TR = corners_img[0], corners_img[1]
    ax.plot([TL[0], TR[0]], [TL[1], TR[1]], linewidth=2, color="yellow",
            label="top edge")

    # Original points
    ax.scatter(points_img[:, 0], points_img[:, 1],
               c="red", s=80, marker="o", edgecolors="white", linewidths=1.5,
               label="original")

    # Projected points on the top edge
    ax.scatter(projs_img[:, 0], projs_img[:, 1],
               c="cyan", s=90, marker="x", linewidths=2,
               label="projection")

    # Connectors + labels
    for i, (p, q) in enumerate(zip(points_img, projs_img)):
        ax.plot([p[0], q[0]], [p[1], q[1]], linestyle="--", linewidth=1.5,
                color="cyan")
        if labels:
            ax.text(p[0] + 5, p[1] - 5, labels[i], color="blue", fontsize=10,
                    weight="bold")
        if dist_mm is not None:
            mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
            ax.text(mx, my, f"{dist_mm[i]:.1f} mm",
                    color="white", fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black",
                              alpha=0.35))

    ax.axis("off")
    ax.legend(loc="lower right")

    return fig


def get_foot_sizes_mm(ref_obj_corners, foot_points, is_vis=False,
                      img_orig=None, img_orig_path=None):
    """Returns length and width of feet in mm based on reference
    rectangular object corners and 3 points of foot."""

    # size_mm = (215.9,279.4)

    size_mm = pick_size_mm_for_top_edge(ref_obj_corners)

    print(size_mm[1])

    # If you don’t have calibration, call without it:
    # Letter; use (210,297) for A4
    projs_img, projs_paper, dist_mm = project_points_to_top_edge(
        ref_obj_corners, foot_points, size_mm=size_mm
    )

    dist_toe_paper_edge = dist_mm[0]
    # sum of dist(toe, top_paper_edge) and vertical paper side length (depends
    # on how it is placed); toe can be about paper edge or below
    foot_length_mm = size_mm[1] - dist_toe_paper_edge

    # After you've already computed:
    # projs_img, projs_paper, dist_mm = project_points_to_top_edge(...)
    # and you have the labels in this order for your 3 points:
    labels = ["toe", "most_left", "most_right"]

    # Find indices
    idx = {lbl: i for i, lbl in enumerate(labels)}
    iL, iR = idx["most_left"], idx["most_right"]

    # ---- Option A: real-world distance in millimeters (recommended) ----
    # projs_paper holds coordinates in the rectified paper plane (mm), so the
    # projections lie on y=0. Distance along the top edge is |Δx|:
    xL, xR = projs_paper[iL, 0], projs_paper[iR, 0]
    dist_edge_mm = float(abs(xR - xL))
    print(f"Distance between projected 'most_left' and 'most_right'\
           (mm): {dist_edge_mm:.2f}")

    # - Option B: pixel distance in the original image (if you prefer pixels)
    pL, pR = projs_img[iL], projs_img[iR]
    dist_edge_px = float(np.linalg.norm(pR - pL))
    print(f"Distance between projected points (pixels): {dist_edge_px:.2f}")

    if is_vis:
        fig = maks_vis(img_orig, ref_obj_corners, foot_points, projs_img)
        fig.savefig(img_orig_path[:-4] + "_vis.jpg", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)

    return foot_length_mm, dist_edge_mm


def get_standardized_length(foot_length_mm: float, gender: str = "f") -> dict:
    """
    Converts foot length in mm to standardized shoe sizes
    (US, UK, EU) using Mondopoint reference tables (ISO 9407:2019)

    Parameters:
        foot_length_mm (float): Foot length in millimeters.
        gender (str): 'men' or 'women'

    Returns:
        dict: {"EU": ..., "US": ..., "UK": ...}
    """
    # Convert mm → cm
    foot_length_cm = foot_length_mm / 10.0

    # Select table
    if gender.lower() == "m":
        table = MEN_TABLE
    elif gender.lower() == "f":
        table = WOMEN_TABLE
    else:
        raise ValueError("Gender must be 'm' or 'f'")

    # Find closest available Mondopoint length in the table
    closest_length = min(table.keys(), key=lambda x: abs(x - foot_length_cm))

    return {"Mondopoint_cm": closest_length, **table[closest_length]}


def get_standardized_width(width_mm: float, gender: str = "f") -> str:
    """
    Convert foot width (mm) into a standard width category (US-style).
    """
    gender = gender.lower()

    if gender == "m":
        table = MEN_WIDTH_TABLE
    elif gender == "f":
        table = WOMEN_WIDTH_TABLE
    else:
        raise ValueError("Gender must be 'm' or 'f'.")

    for category, (low, high) in table.items():
        if low <= width_mm <= high:
            return category

    return "Unknown"


def get_foot_size_estm(img_path, gender='f', ref_obj='paper_letter',
                       is_wall=True):
    """Returns length and width in mm of foot based on input image.
    There are 3 options that are accepted as reference size object:
    1) Letter paper(default option); 2) A4 paper; 3) plastic card.
    For higher precision heel can "touch" touch wall  - so, it will be
    a straight line from which "feet silueht" starts.
    """

    # Load a model;
    # todo: should be done as class to avoid init on each inference
    model = SAM("sam2.1_t.pt")

    results_foot = model(img_path, points=[[[1683, 1530]]], labels=[[1]])
    results_paper = model(img_path, points=[[[178, 1906], [2755, 1989]]],
                         labels=[[1, 1]])
    # print(type(results))

    # should be picked by some reference points (e.g. colour for paper or
    # position for feet)
    img_orig = np.array(results_paper[0].orig_img)
    masks = results_paper[0].masks.data
    masks = masks.cpu().numpy()
    paper_mask = masks[0]

    masks = results_foot[0].masks.data
    masks = masks.cpu().numpy()
    foot_mask = masks[0]

    ref_obj_corners = get_ref_obj_corners(paper_mask)
    # how to define width if is_wall = False
    foot_points = get_foot_points(foot_mask)

    feet_length_mm, feet_width_mm = get_foot_sizes_mm(ref_obj_corners,
                                                      foot_points, is_vis=True,
                                                      img_orig=img_orig,
                                                      img_orig_path=img_path)

    feet_length_stnd = get_standardized_length(feet_length_mm, gender="m")
    feet_width_stnd = get_standardized_width(feet_width_mm, gender="m")

    # logging
    print("")
    print(f"foot length: {feet_length_mm:.2f} mm")
    print(f"foot width: {feet_width_mm:.2f} mm")
    print(f"foot size: US {feet_length_stnd['US']};\
          EU {feet_length_stnd['EU']};\
          UK {feet_length_stnd['UK']}")
    print(f"foot width: {feet_width_stnd}")
    print("")
    print(f"image with size visualisations is stored at:\
          {img_path[:-4] + '_vis.jpg'}")
    print("")

    # feet_length_mm = 269.12
    # feet_width_mm = 99.12

    # return feet_length_mm, len(results)
    return feet_length_mm, feet_width_mm  # , img_vis


if __name__ == "__main__":
    img_path = "./foot_size_estm/IMG_7350.jpg"

    feet_length_mm, feet_width_mm = get_foot_size_estm(img_path)

    print("")
    print(feet_length_mm, feet_width_mm)
