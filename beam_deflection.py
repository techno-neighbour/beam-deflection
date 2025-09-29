import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math

support_points = []
curve_points = []
calib_points = []
image_to_show = None
original_image = None
MAGNIFIER_SIZE = 150 # Size of the magnifier window
ZOOM_FACTOR = 4      # How much to zoom in

def mouse_event(event, x, y, flags, param):
    """to handle both mouse clicks and movement for the magnifier."""
    global image_to_show, original_image, support_points, curve_points

    if event == cv2.EVENT_MOUSEMOVE:
        x_start = max(0, x - MAGNIFIER_SIZE // (2 * ZOOM_FACTOR))
        y_start = max(0, y - MAGNIFIER_SIZE // (2 * ZOOM_FACTOR))
        x_end = min(original_image.shape[1], x + MAGNIFIER_SIZE // (2 * ZOOM_FACTOR))
        y_end = min(original_image.shape[0], y + MAGNIFIER_SIZE // (2 * ZOOM_FACTOR))
        roi = original_image[y_start:y_end, x_start:x_end]
        magnified_roi = cv2.resize(roi, (MAGNIFIER_SIZE, MAGNIFIER_SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.line(magnified_roi, (MAGNIFIER_SIZE//2, 0), (MAGNIFIER_SIZE//2, MAGNIFIER_SIZE), (0, 0, 255), 1)
        cv2.line(magnified_roi, (0, MAGNIFIER_SIZE//2), (MAGNIFIER_SIZE, MAGNIFIER_SIZE//2), (0, 0, 255), 1)
        cv2.imshow("Magnifier", magnified_roi)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(support_points) < 2:
            support_points.append((x, y))
            print(f"Captured support point #{len(support_points)}.")
            cv2.circle(image_to_show, (x, y), 7, (0, 255, 0), -1)
        else:
            curve_points.append((x, y))
            cv2.circle(image_to_show, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image_to_show)

def calib_click_event(event, x, y, flags, param):
    """to separate callback for calibration clicks."""
    global calib_points
    if event == cv2.EVENT_LBUTTONDOWN and len(calib_points) < 2:
        print(f"Captured calibration point #{len(calib_points) + 1}.")
        calib_points.append((x, y))
        cv2.circle(image_to_show, (x, y), 7, (255, 165, 0), -1)
        cv2.imshow("Calibration", image_to_show)

def get_manual_deflection():
    """for user to enter beam properties and calculates theoretical deflection."""
    print("\n--- Manual Deflection Calculation ---")
    try:
        P = float(input("Enter Load (P) in Newtons (N): "))
        L = float(input("Enter Beam Length (L) between supports in meters (m): "))
        E = float(input("Enter Modulus of Elasticity (E) in GigaPascals (GPa): ")) * 1e9
        b = float(input("Enter cross-section base (b) in meters (m): "))
        h = float(input("Enter cross-section height (h) in meters (m): "))
        I = (b * h**3) / 12
        deflection_manual = (P * L**3) / (48 * E * I)
        print(f"Theoretical Max Deflection: {deflection_manual:.6f} meters")
        return deflection_manual
    except ValueError:
        print("Invalid input. Skipping manual calculation.")
        return None

def plot_deflection_graph(baseline_func, curve_func, x_range, m_per_px):
    """calculates deflection and generates a graph in matplotlib with a dual y-axis."""
    x_smooth = np.linspace(x_range[0], x_range[1], num=500)
    y_baseline = baseline_func(x_smooth)
    y_curve = curve_func(x_smooth)
    deflection_px = y_curve - y_baseline
    
    max_deflection_px = np.max(deflection_px)
    max_deflection_m = max_deflection_px * m_per_px
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # plot pixels on the left axis (ax1)
    ax1.plot(x_smooth, deflection_px, color='blue', label='Deflection Curve')
    ax1.set_xlabel("Position Along Beam (pixels)", fontsize=12)
    ax1.set_ylabel("Deflection (pixels)", color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # second y-axis (ax2) sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Deflection (meters)", color='green', fontsize=12)
    
    # set limits of the second axis to correspond to the first
    px_min, px_max = ax1.get_ylim()
    ax2.set_ylim(px_min * m_per_px, px_max * m_per_px)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.invert_yaxis()

    # mark max deflection
    max_idx = np.argmax(deflection_px)
    label_text = f'Max: {max_deflection_px:.2f} px ({max_deflection_m*1000:.2f} mm)'
    ax1.scatter(x_smooth[max_idx], max_deflection_px, color='red', zorder=5, label=label_text)
    ax1.legend(loc='upper left')
    
    fig.suptitle("Beam Deflection Curve", fontsize=16)
    fig.tight_layout()
    plt.show()
    
    return max_deflection_px

if __name__ == "__main__":
    deflection_manual_m = None
    if input("Perform manual formula-based calculation? (y/n): ").lower() == 'y':
        deflection_manual_m = get_manual_deflection()

    img_folder = './BeamImages/'
    image_paths = sorted(glob.glob(img_folder + '*.png') + glob.glob(img_folder + '*.jpg'))
    if not image_paths:
        raise FileNotFoundError(f"No images found in '{img_folder}'!")

    for img_path in image_paths:
        print(f"\n--- Processing image: {img_path} ---")
        support_points, curve_points, calib_points = [], [], []

        original_image = cv2.imread(img_path)
        if original_image is None: continue
        
        image_to_show = original_image.copy()
        cv2.namedWindow("Image")
        cv2.namedWindow("Magnifier")
        cv2.setMouseCallback("Image", mouse_event)
        
        print("1. Use the 'Magnifier' for precision.")
        print("2. Click LEFT & RIGHT supports, then trace the BENT edge.")
        print("3. Press 'c' to calculate, 'q' to quit.")

        while True:
            cv2.imshow("Image", image_to_show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(support_points) < 2 or len(curve_points) < 3:
                    print("Error: Select 2 supports and at least 3 curve points.")
                    continue
                break
            if key == ord('q'): exit("Exiting program.")
        cv2.destroyAllWindows()

        # sort the curve points by x-coordinate
        curve_points = np.array(sorted(curve_points, key=lambda p: p[0]))
        
        # --- fix for duplicate x-values ---
        _, unique_indices = np.unique(curve_points[:, 0], return_index=True)
        if len(unique_indices) < len(curve_points):
            print("Warning: Duplicate x-values detected in clicks. Using only the first instance of each.")
            curve_points = curve_points[unique_indices]
        
        # --- make interpolation robust ---
        kind = 'cubic' if len(curve_points) >= 4 else 'linear'
        
        supports = np.array(support_points)
        curve_func = interp1d(curve_points[:,0], curve_points[:,1], kind=kind, fill_value="extrapolate")
        baseline_func = interp1d(supports[:,0], supports[:,1], kind='linear', fill_value="extrapolate")
        x_range = (curve_points[:,0].min(), curve_points[:,0].max())
        
        # --- Calibration ---
        meters_per_pixel = None
        if input("Calibrate to meters with a reference object? (y/n): ").lower() == 'y':
            # user calibration
            print("Click the TWO ENDPOINTS of your reference object.")
            image_to_show = original_image.copy()
            cv2.namedWindow("Calibration")
            cv2.setMouseCallback("Calibration", calib_click_event)
            while len(calib_points) < 2:
                cv2.imshow("Calibration", image_to_show)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cv2.destroyAllWindows()
            if len(calib_points) == 2:
                try:
                    known_length = float(input("Enter reference length in METERS: "))
                    pixel_dist = math.dist(calib_points[0], calib_points[1])
                    meters_per_pixel = known_length / pixel_dist
                except (ValueError, ZeroDivisionError) as e:
                    print(f"Calibration failed: {e}. Using default estimate.")

        if meters_per_pixel is None:
            # default calibration
            print("Using default 'rough estimate' for scaling.")
            print("ASSUMPTION: The beam's length between supports is 1.0 meters.")
            beam_length_px = math.dist(support_points[0], support_points[1])
            meters_per_pixel = 1.0 / beam_length_px
        
        # --- Final Output ---
        print("Generating deflection graph...")
        max_deflection_px = plot_deflection_graph(baseline_func, curve_func, x_range, meters_per_pixel)
        deflection_measured_m = max_deflection_px * meters_per_pixel
        
        print("\n--- FINAL RESULTS ---")
        print(f"Scaling Factor: {meters_per_pixel:.6f} meters/pixel")
        print(f"Max Measured Deflection: {deflection_measured_m:.6f} meters ({deflection_measured_m*1000:.2f} mm)")

        if deflection_manual_m is not None:
            error = abs((deflection_measured_m - deflection_manual_m) / deflection_manual_m) * 100
            print("\n--- ACCURACY COMPARISON ---")
            print(f"Theoretical Deflection: {deflection_manual_m:.6f} m")
            print(f"Measured Deflection:    {deflection_measured_m:.6f} m")
            print(f"Error: {error:.2f}%")
            
    print("\nProcessing complete ✅")