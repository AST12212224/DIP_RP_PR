"""
Pixel Rearrangement Art Generator with ENHANCED ANIMATION
Digital Image Processing Mini Project

Enhanced features:
- Extended video duration (3s source → 4s animation → 3s result = 10s total)
- User input for image paths with fallback defaults
- Smooth transitions and professional video output
"""

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
Path("output").mkdir(exist_ok=True)

# ==================== CORE ALGORITHM WITH ENHANCED ANIMATION ====================


def pixel_rearrangement_art_animated(
    source_path, destination_path, output_path, num_frames=120, show_live=True
):
    """
    Main function with ENHANCED ANIMATION:
    - Shows source image for 3 seconds
    - Animates transition for 6 seconds (SLOW with MORE frames)
    - Shows final result for 3 seconds
    Total video duration: 12 seconds

    Parameters:
    - num_frames: Number of animation frames for the transition (120 = very smooth)
    - show_live: If True, shows live animation window
    """
    print(f"\n{'=' * 60}")
    print(f"🎬 ANIMATING: {Path(source_path).name} → {Path(destination_path).name}")
    print(f"{'=' * 60}")

    # Load images
    source = cv2.imread(source_path)
    destination = cv2.imread(destination_path)

    if source is None or destination is None:
        print(f"❌ Error loading images!")
        return None

    # Resize both to same dimensions
    height, width = 512, 512
    source = cv2.resize(source, (width, height))
    destination = cv2.resize(destination, (width, height))

    # Convert to grayscale for intensity calculation
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    dest_gray = cv2.cvtColor(destination, cv2.COLOR_BGR2GRAY)

    print("📊 Analyzing pixel intensities...")

    # Flatten images and get pixel data
    h, w = source_gray.shape
    total_pixels = h * w

    # Create arrays with (grayscale_value, original_position, rgb_color)
    source_pixels = []
    for i in range(h):
        for j in range(w):
            gray_val = source_gray[i, j]
            rgb_color = source[i, j]
            source_pixels.append((gray_val, (i, j), rgb_color))

    # Create array with (grayscale_value, destination_position)
    dest_positions = []
    for i in range(h):
        for j in range(w):
            gray_val = dest_gray[i, j]
            dest_positions.append((gray_val, (i, j)))

    print("🔄 Sorting pixels by intensity...")

    # Sort both by grayscale intensity
    source_pixels.sort(key=lambda x: x[0])
    dest_positions.sort(key=lambda x: x[0])

    # Create mapping: source_position → destination_position
    pixel_mapping = []
    for idx in range(total_pixels):
        gray_val, source_pos, rgb_color = source_pixels[idx]
        dest_gray_val, dest_pos = dest_positions[idx]

        pixel_mapping.append(
            (
                source_pos[0],
                source_pos[1],  # Start position
                dest_pos[0],
                dest_pos[1],  # End position
                rgb_color,  # Color to maintain
            )
        )

    print(f"🎨 Generating enhanced video with extended duration...")

    # Video parameters
    fps = 30

    # Phase durations (in seconds)
    source_duration = 3  # Show source for 3 seconds
    animation_duration = 6  # Animate for 6 seconds (SLOW & SMOOTH)
    result_duration = 3  # Show result for 3 seconds

    # Calculate frame counts
    source_frames_count = source_duration * fps
    animation_frames_count = num_frames
    result_frames_count = result_duration * fps

    all_frames = []

    # Create live display window if requested
    if show_live:
        cv2.namedWindow("Pixel Rearrangement Animation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pixel Rearrangement Animation", 800, 800)

    # ========== PHASE 1: Source Image (3 seconds) ==========
    print(f"  Phase 1/3: Displaying source image ({source_duration}s)...")
    for i in range(source_frames_count):
        frame = source.copy()

        # Add text overlay
        cv2.putText(
            frame,
            "Source Image",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        all_frames.append(frame.copy())

        if show_live and i % 10 == 0:
            cv2.imshow("Pixel Rearrangement Animation", frame)
            cv2.waitKey(1)

    # ========== PHASE 2: Animation Transition (6 seconds) ==========
    print(
        f"  Phase 2/3: Animating transition ({animation_duration}s, {num_frames} frames)..."
    )
    print(f"    🐌 Slow transformation for mesmerizing effect...")

    # Calculate how many times to repeat each animation frame to fill 6 seconds
    total_animation_frames_needed = animation_duration * fps  # 6s * 30fps = 180 frames
    frames_per_step = max(
        1, total_animation_frames_needed // num_frames
    )  # Repeat each frame

    print(
        f"    📊 Rendering {num_frames} unique positions, each held for {frames_per_step} frames"
    )

    for frame_num in range(num_frames + 1):
        # Calculate interpolation factor (0.0 to 1.0)
        t = frame_num / num_frames

        # Use easing function for smoother animation (ease-in-out cubic)
        t_eased = t * t * (3.0 - 2.0 * t)

        # Create current frame
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Interpolate each pixel position
        for start_y, start_x, end_y, end_x, color in pixel_mapping:
            # Linear interpolation between start and end positions
            current_y = int(start_y + (end_y - start_y) * t_eased)
            current_x = int(start_x + (end_x - start_x) * t_eased)

            # Ensure within bounds
            current_y = max(0, min(h - 1, current_y))
            current_x = max(0, min(w - 1, current_x))

            # Place pixel at interpolated position
            frame[current_y, current_x] = color

        # Add progress text on frame
        progress_text = f"Transforming: {int(t * 100)}%"
        cv2.putText(
            frame,
            progress_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Add this frame multiple times to slow down the animation
        for repeat in range(frames_per_step):
            all_frames.append(frame.copy())

        # Show live animation
        if show_live:
            cv2.imshow("Pixel Rearrangement Animation", frame)
            cv2.waitKey(1)

        # Print progress
        if frame_num % 10 == 0 or frame_num == num_frames:
            print(f"    Animation step {frame_num}/{num_frames} ({int(t * 100)}%)")

    # ========== PHASE 3: Final Result (3 seconds) ==========
    print(f"  Phase 3/3: Displaying final result ({result_duration}s)...")

    final_frame = all_frames[-1].copy()

    # Remove progress text for clean final frames
    final_frame_clean = frame.copy()

    for i in range(result_frames_count):
        result_frame = final_frame_clean.copy()

        # Add completion text
        cv2.putText(
            result_frame,
            "Final Result",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        all_frames.append(result_frame.copy())

        if show_live and i % 10 == 0:
            cv2.imshow("Pixel Rearrangement Animation", result_frame)
            cv2.waitKey(1)

    if show_live:
        cv2.destroyAllWindows()

    # ========== Save Video ==========
    print("💾 Saving enhanced video...")

    video_path = output_path.replace(".jpg", ".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for frame in all_frames:
        video.write(frame)

    video.release()

    total_duration = source_duration + animation_duration + result_duration
    print(f"✅ Video saved: {video_path}")
    print(
        f"   Duration: {total_duration} seconds (3s source + 6s transform + 3s result)"
    )
    print(f"   Total frames: {len(all_frames)} @ {fps} fps")

    # Save key frames as images
    print("💾 Saving key frames...")

    cv2.imwrite(output_path.replace(".jpg", "_start.jpg"), all_frames[0])
    cv2.imwrite(
        output_path.replace(".jpg", "_middle.jpg"),
        all_frames[source_frames_count + num_frames // 2],
    )
    cv2.imwrite(output_path.replace(".jpg", "_final.jpg"), all_frames[-1])

    # Create comparison grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Source Image\n(Original)", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(destination, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(
        "Destination Template\n(Target Structure)", fontsize=14, fontweight="bold"
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(final_frame_clean, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(
        "Final Result\n(After Rearrangement)", fontsize=14, fontweight="bold"
    )
    axes[0, 2].axis("off")

    # Show animation progression
    axes[1, 0].imshow(cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Start (0s)", fontsize=12)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        cv2.cvtColor(
            all_frames[source_frames_count + num_frames // 2], cv2.COLOR_BGR2RGB
        )
    )
    axes[1, 1].set_title(
        f"Mid-Transformation (~{source_duration + animation_duration / 2:.1f}s)",
        fontsize=12,
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(cv2.cvtColor(all_frames[-1], cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f"End ({total_duration}s)", fontsize=12)
    axes[1, 2].axis("off")

    plt.suptitle(
        f"Pixel Rearrangement Animation - {total_duration}s Video (6s Slow Transform)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    comparison_path = output_path.replace(".jpg", "_comparison.jpg")
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"✅ Comparison saved: {comparison_path}")
    print(f"✅ Animation complete! Check output folder for:")
    print(f"   📹 Video: {Path(video_path).name} ({total_duration}s)")
    print(f"   🖼️  Images: *_start.jpg, *_middle.jpg, *_final.jpg")
    print(f"   📊 Comparison: {Path(comparison_path).name}\n")

    return final_frame_clean


def pixel_rearrangement_art_fast(source_path, destination_path, output_path):
    """
    Fast version without animation - just creates final result
    Use this for quick results or for Task 4.2 and 4.3
    """
    source = cv2.imread(source_path)
    destination = cv2.imread(destination_path)

    if source is None or destination is None:
        print(f"❌ Error loading images!")
        return None

    height, width = 512, 512
    source = cv2.resize(source, (width, height))
    destination = cv2.resize(destination, (width, height))

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    dest_gray = cv2.cvtColor(destination, cv2.COLOR_BGR2GRAY)

    h, w = source_gray.shape
    total_pixels = h * w

    source_pixels = []
    for i in range(h):
        for j in range(w):
            gray_val = source_gray[i, j]
            rgb_color = source[i, j]
            source_pixels.append((gray_val, (i, j), rgb_color))

    dest_positions = []
    for i in range(h):
        for j in range(w):
            gray_val = dest_gray[i, j]
            dest_positions.append((gray_val, (i, j)))

    source_pixels.sort(key=lambda x: x[0])
    dest_positions.sort(key=lambda x: x[0])

    output = np.zeros_like(source)

    for idx in range(total_pixels):
        source_gray_val, source_pos, source_rgb = source_pixels[idx]
        dest_gray_val, dest_pos = dest_positions[idx]
        output[dest_pos[0], dest_pos[1]] = source_rgb

    cv2.imwrite(output_path, output)
    print(f"✓ Fast mode: {output_path}")

    return output


# ==================== USER INPUT FUNCTION ====================


def get_user_input():
    """
    Get image paths from user with intelligent defaults
    Returns: (source_path, destination_path, output_path)
    """
    print("\n" + "=" * 60)
    print("🎨 PIXEL REARRANGEMENT ART GENERATOR")
    print("=" * 60)
    print("\nEnter image paths (or press Enter to use defaults)")
    print("-" * 60)

    # Get source image
    source_input = input("Source image path [default: images/bhayia.jpg]: ").strip()
    source_path = source_input if source_input else "images/bhayia.jpg"

    # Get destination image
    dest_input = input("Destination image path [default: images/vahini.jpg]: ").strip()
    destination_path = dest_input if dest_input else "images/vahini.jpg"

    # Get output name (without extension)
    output_input = input(
        "Output filename (without .jpg) [default: art_output]: "
    ).strip()
    output_name = output_input if output_input else "art_output"
    output_path = f"output/{output_name}.jpg"

    print("\n" + "=" * 60)
    print("📁 Configuration:")
    print(f"   Source: {source_path}")
    print(f"   Destination: {destination_path}")
    print(f"   Output: {output_path}")
    print("=" * 60)

    # Verify files exist
    if not Path(source_path).exists():
        print(f"⚠️  Warning: Source file not found: {source_path}")
        print("   Using default: images/bhayia.jpg")
        source_path = "images/bhayia.jpg"

    if not Path(destination_path).exists():
        print(f"⚠️  Warning: Destination file not found: {destination_path}")
        print("   Using default: images/vahini.jpg")
        destination_path = "images/vahini.jpg"

    return source_path, destination_path, output_path


# ==================== TASK 4.2: CONTOUR DETECTION ====================


def contour_detection_analysis(image_paths):
    """
    Task 4.2: Perform contour finding with different approximation methods
    """
    print("\n" + "=" * 60)
    print("TASK 4.2: CONTOUR DETECTION ANALYSIS")
    print("=" * 60)

    colors = [
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 0),  # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {img_path}")
            continue

        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(gray, 50, 150)

        contours_none, hierarchy_none = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        contours_simple, hierarchy_simple = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        img_none = img.copy()
        img_simple = img.copy()
        img_colored = img.copy()
        img_hierarchy = img.copy()

        cv2.drawContours(img_none, contours_none, -1, (0, 255, 0), 2)
        cv2.drawContours(img_simple, contours_simple, -1, (255, 0, 0), 2)

        for i, contour in enumerate(contours_simple[:5]):
            color = colors[i % len(colors)]
            cv2.drawContours(img_colored, [contour], -1, color, 2)

        if hierarchy_none is not None:
            for i, contour in enumerate(contours_none):
                if cv2.contourArea(contour) > 100:
                    h = hierarchy_none[0][i]
                    if h[3] == -1:
                        color = (0, 0, 255)
                    elif h[2] == -1:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    cv2.drawContours(img_hierarchy, [contour], -1, color, 2)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Original Image {idx + 1}")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(edges, cmap="gray")
        axes[0, 1].set_title("Canny Edges")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(cv2.cvtColor(img_none, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f"CHAIN_APPROX_NONE\n({len(contours_none)} contours)")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(cv2.cvtColor(img_simple, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(f"CHAIN_APPROX_SIMPLE\n({len(contours_simple)} contours)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Multi-Color Contours")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(cv2.cvtColor(img_hierarchy, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title("Hierarchy\nRed=Outer, Green=Inner, Blue=Nested")
        axes[1, 2].axis("off")

        plt.tight_layout()
        output_file = f"output/task_4.2_contours_image_{idx + 1}.jpg"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"✓ Image {idx + 1}: Found {len(contours_simple)} contours")


# ==================== TASK 4.3: HARRIS CORNER DETECTION ====================


def harris_corner_detection(image_paths):
    """
    Task 4.3: Perform Harris corner detection
    """
    print("\n" + "=" * 60)
    print("TASK 4.3: HARRIS CORNER DETECTION")
    print("=" * 60)

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load {img_path}")
            continue

        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        dst = cv2.dilate(dst, None)

        img_corners = img.copy()
        img_corners[dst > 0.01 * dst.max()] = [0, 0, 255]

        img_circles = img.copy()
        corner_coords = np.where(dst > 0.01 * dst.max())
        corners = list(zip(corner_coords[1], corner_coords[0]))

        for corner in corners:
            cv2.circle(img_circles, corner, 3, (0, 255, 0), -1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original Image {idx + 1}")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Harris Corners (Red)\n{len(corners)} corners detected")
        axes[1].axis("off")

        axes[2].imshow(cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f"Corner Points (Green Circles)")
        axes[2].axis("off")

        plt.tight_layout()
        output_file = f"output/task_4.3_harris_image_{idx + 1}.jpg"
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"✓ Image {idx + 1}: Detected {len(corners)} Harris corners")


# ==================== MAIN EXECUTION ====================


def main():
    """
    Main execution with user input
    """
    # Get paths from user (or use defaults)
    source_path, destination_path, output_path = get_user_input()

    # Run animation
    pixel_rearrangement_art_animated(
        source_path=source_path,
        destination_path=destination_path,
        output_path=output_path,
        num_frames=120,  # More frames = smoother, slower transformation
        show_live=True,
    )


if __name__ == "__main__":
    main()
