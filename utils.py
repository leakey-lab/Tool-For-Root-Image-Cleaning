from blur_detector import (
    LaplacianBlurDetector,
    compute_and_store_blur_scores,
    calculate_global_statistics,
)

from dash import no_update
import os
import torch
import dash

detector = LaplacianBlurDetector().eval()
if torch.cuda.is_available():
    detector = detector.cuda()


def fetch_and_check_blurry(
    image_path, blur_threshold, blur_scores, mean_blur, std_blur
):

    blur_score = blur_scores.get(os.path.normpath(image_path))
    lower_bound = mean_blur - blur_threshold * std_blur
    is_blurry = blur_score < lower_bound
    if is_blurry:
        return image_path, blur_score
    return None


def handle_blurry_deletion(
    filtered_blurry_images, selected_values, global_blur_stats, blur_detection_state
):
    if not filtered_blurry_images or not global_blur_stats:
        return no_update, no_update, "", no_update

    # Identify the selected images to delete
    selected_indices = [i for i, val in enumerate(selected_values) if "checked" in val]
    images_to_delete = [filtered_blurry_images[i] for i in selected_indices]

    # Update blurred_images list by removing the selected images
    updated_blurred_images = [
        img for i, img in enumerate(filtered_blurry_images) if i not in selected_indices
    ]

    # Update global_blur_stats by removing the blur scores of the deleted images
    updated_blur_scores = {
        k: v
        for k, v in global_blur_stats["blur_scores"].items()
        if k not in images_to_delete
    }

    # Recalculate mean and standard deviation for the updated blur scores
    if updated_blur_scores:
        mean_blur = sum(updated_blur_scores.values()) / len(updated_blur_scores)
        std_blur = (
            sum((x - mean_blur) ** 2 for x in updated_blur_scores.values())
            / len(updated_blur_scores)
        ) ** 0.5
    else:
        mean_blur = 0
        std_blur = 0

    updated_global_blur_stats = {
        "blur_scores": updated_blur_scores,
        "mean_blur": mean_blur,
        "std_blur": std_blur,
    }

    # Delete the selected images from the filesystem
    for image_path in images_to_delete:
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except OSError as e:
            print(f"Error deleting {image_path}: {e}")

    return (
        blur_detection_state,
        [updated_blurred_images, []],  # We don't have updated blur values here
        "Deletion completed",
        updated_global_blur_stats,
    )


def handle_blur_detection(folder_data, blur_detection_state, blurred_images):
    if not folder_data or "path" not in folder_data:
        return blur_detection_state, blurred_images, "", None

    folder_path = folder_data["path"]
    ctx = dash.callback_context
    if "select-folder-blur" in ctx.triggered[0]["prop_id"]:
        blur_detection_state = {"running": True, "completed": False, "progress": 0}
        blurred_images = []

    if not blur_detection_state["running"]:
        return blur_detection_state, blurred_images, "", None

    total_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    file_count = len(total_files)

    if blur_detection_state["completed"]:
        return blur_detection_state, blurred_images, "", None

    new_blurred_images, blur_val = [], []
    blur_scores_global = compute_and_store_blur_scores(total_files, detector)
    mean_blur_global, std_blur_global = calculate_global_statistics(blur_scores_global)

    for i, file_path in enumerate(total_files):
        result = fetch_and_check_blurry(
            file_path, 1, blur_scores_global, mean_blur_global, std_blur_global
        )
        if result is not None:
            new_blurred_images.append(file_path)
            blur_val.append(result[1])

        blur_detection_state["progress"] = int(((i + 1) / file_count) * 100)

    blur_detection_state["completed"] = True
    blur_stats_data = {
        "blur_scores": blur_scores_global,
        "mean_blur": mean_blur_global,
        "std_blur": std_blur_global,
    }
    return blur_detection_state, [new_blurred_images, blur_val], "", blur_stats_data
