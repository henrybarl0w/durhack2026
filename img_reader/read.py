import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION: Choose which images to display ---
# 1: Original Grayscale
# 2: Just Horizontal Lines
# 3: Cleaned Blotches
# 4: 50% Dark-Density Circles + Blue Lines
IMAGES_TO_PRINT = [1, 2, 3, 4]

# 1. Load the image in grayscale (0 to 255)
gray_img = cv2.imread("score.png", cv2.IMREAD_GRAYSCALE)
if gray_img is None:
    raise FileNotFoundError("Could not load 'score.png'. Please check the file path.")

height, width = gray_img.shape

# --- PART A: Extract JUST the Horizontal Lines ---
row_means = np.mean(gray_img, axis=1)
global_mean = np.mean(row_means)

window = 3
line_candidates = []
for y in range(window, height - window):
    segment = row_means[y - window : y + window + 1]
    if row_means[y] == np.min(segment) and row_means[y] < (global_mean * 0.85):
        line_candidates.append(y)

refined_lines = []
for y in line_candidates:
    best_y = y
    min_pixel_sum = float('inf')
    for dy in range(-2, 3):
        ny = y + dy
        if 0 <= ny < height:
            row_sum = np.sum(gray_img[ny, :])
            if row_sum < min_pixel_sum:
                min_pixel_sum = row_sum
                best_y = ny
    if best_y not in refined_lines:
        refined_lines.append(best_y)

refined_lines.sort()

# Create image of just lines (White background = 255, Lines = 0)
just_lines_img = np.full_like(gray_img, 255)
for y in refined_lines:
    just_lines_img[y, :] = 0


# --- PART B: Extract & Clean Dark/Black Blotches (Notes & Symbols) ---
_, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
detected_staves = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
blotches_thresh = cv2.subtract(thresh, detected_staves)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blotches_thresh, connectivity=8)
cleaned_blotches_thresh = np.zeros_like(blotches_thresh)

valid_blotches = []
for i in range(1, num_labels):
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    
    # Eliminate thin vertical lines/artifacts
    if w <= 3 and h > 10:
        continue
        
    blotch_length = max(w, h)
    blotch_width = min(w, h)
    if blotch_length > 3 * blotch_width and blotch_width < 5:
        continue
        
    cleaned_blotches_thresh[labels == i] = 255
    valid_blotches.append(i)

blotches_img = cv2.bitwise_not(cleaned_blotches_thresh)


# --- PART C: Grow Circles Until 50% Dark Pixel Density ---
# Define dark pixels threshold (pixels darker than background)
dark_threshold = 128
y_indices, x_indices = np.ogrid[:height, :width]

grown_circles_img = cv2.cvtColor(blotches_img, cv2.COLOR_GRAY2RGB)

# Calculate average line spacing for tolerance threshold
if len(refined_lines) > 1:
    line_spacings = np.diff(refined_lines)
    avg_spacing = np.median(line_spacings)
else:
    avg_spacing = 10

tolerance = avg_spacing * 0.35

print(f"{'Blotch ID':<10} | {'Center (X, Y)':<18} | {'Grown Radius':<15} | {'Classification':<15}")
print("-" * 65)

circle_data = []

for idx, i in enumerate(valid_blotches, start=1):
    cx, cy = centroids[i]
    cx_int, cy_int = int(cx), int(cy)
    
    # Precompute distance from this blotch center to all pixels
    dist_from_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
    
    # Grow circle unit by unit starting from r = 2
    best_r = 2
    for r in range(2, 40):
        circle_mask = (dist_from_center <= r)
        total_pixels = np.sum(circle_mask)
        if total_pixels == 0:
            continue
        
        # Count dark pixels inside the circle
        dark_pixels = np.sum((gray_img < dark_threshold) & circle_mask)
        dark_ratio = dark_pixels / total_pixels
        
        best_r = r
        # Stop growing when dark pixel density drops to 50% or below
        if dark_ratio <= 0.50:
            break
            
    # Classify position based on center Y relative to blue lines
    lines_touched = [l for l in refined_lines if (cy_int - best_r - 1) <= l <= (cy_int + best_r + 1)]
    
    if len(lines_touched) >= 2:
        position = "Between Lines"
        circle_color = [255, 140, 0]  # Orange
    else:
        if refined_lines:
            closest_line = min(refined_lines, key=lambda l: abs(l - cy))
            distance = abs(cy - closest_line)
            if distance <= tolerance:
                position = "On Line"
                circle_color = [0, 255, 0]  # Green
            else:
                position = "Between Lines"
                circle_color = [255, 140, 0]  # Orange
        else:
            position = "Unknown"
            circle_color = [128, 128, 128]

    print(f"{i:<10} | ({cx_int}, {cy_int}){'':<6} | {best_r:<15} | {position:<15}")
    
    # Draw the grown circle and center point (Labels removed)
    cv2.circle(grown_circles_img, (cx_int, cy_int), best_r, circle_color, 1)
    cv2.circle(grown_circles_img, (cx_int, cy_int), 2, [0, 0, 0], -1)
    
    circle_data.append({'id': i, 'center': (cx_int, cy_int), 'radius': best_r, 'position': position})

# Overlay the Blue Lines on top of the final image
line_mask = (just_lines_img == 0)
grown_circles_img[line_mask] = [0, 0, 255]


# --- PART D: Display Selected Images ---
if IMAGES_TO_PRINT:
    num_plots = len(IMAGES_TO_PRINT)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    if num_plots == 1:
        axes = [axes]
        
    image_dict = {
        1: (gray_img, 'gray', '1. Original Grayscale'),
        2: (just_lines_img, 'gray', '2. Just Horizontal Lines'),
        3: (blotches_img, 'gray', '3. Cleaned Blotches'),
        4: (grown_circles_img, None, '4. 50% Dark-Density Circles + Blue Lines')
    }
    
    for ax, img_id in zip(axes, IMAGES_TO_PRINT):
        if img_id in image_dict:
            img, cmap, title = image_dict[img_id]
            if cmap:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
            
    plt.tight_layout()
    plt.show()