
#Ai used to help write data validation functions
import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    karts = info_path.get('karts', [])
    detections_all_views = info_path.get('detections', [])

    if not (0 <= view_index < len(detections_all_views)):
        return []

    detections = detections_all_views[view_index]
    karts_list = {}
    id_to_coords = {}

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    image_center = np.array([img_width/2, img_height/2])

    for kart_detection in detections:
        class_id, track_id, x1, y1, x2, y2 = kart_detection
        if class_id != 1:
            continue

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        if (x2_scaled - x1_scaled) < 5 or (y2_scaled - y1_scaled) < 5:
            continue
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2

        if track_id not in id_to_coords:
            id_to_coords[track_id] = {'x': [], 'y': []}
            karts_list[track_id] = {
                "instance_id": track_id,
                "kart_name": karts[track_id],
                "is_center_kart": False,
                "center": (center_x, center_y)
            }

        id_to_coords[track_id]['x'].append(center_x)
        id_to_coords[track_id]['y'].append(center_y)

    ego_id = None
    min_dist = float('inf')
    for track_id in id_to_coords:
        avg_x = np.mean(id_to_coords[track_id]['x'])
        avg_y = np.mean(id_to_coords[track_id]['y'])
        kart_center = (avg_x, avg_y)
        karts_list[track_id]["center"] = kart_center

        dist = np.linalg.norm(np.array(kart_center) - image_center)
        if dist < min_dist:
            min_dist = dist
            ego_id = track_id

    if ego_id is not None:
        karts_list[ego_id]["is_center_kart"] = True

    return list(karts_list.values())



def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    track = info_path.get('track')

    return track


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?


    with open(info_path, 'r') as f:
        file = json.load(f)

    track_name = extract_track_info(file)
    karts = extract_kart_objects(file, view_index, img_width, img_height)

    ego_kart_name = None
    ego_kart_center = [0, 0]

    for kart in karts:
        if kart.get("is_center_kart", False):
            ego_kart_name = str(kart.get("kart_name", "unknown"))
            center = kart.get("center", [0, 0])
            ego_kart_center = [float(center[0]), float(center[1])]
            break

    left_count = 0
    front_count = 0
    qa_pairs = []

    for kart in karts:
        kart_name = str(kart.get("kart_name", "unknown"))
        if kart_name == ego_kart_name:
            continue

        coord = kart.get("center")
        x, y = float(coord[0]), float(coord[1])

        offset_x = int(x - ego_kart_center[0])
        offset_y = int(y - ego_kart_center[1])

        LorR = "right" if offset_x > 0 else "left"
        ForB = "front" if offset_y < 0 else "back"

        if LorR == "left":
            left_count += 1
        if ForB == "front":
            front_count += 1

        relative_position = f"{ForB} and {LorR}"

        qa_pairs.extend([
            {"question": f"Is {kart_name} to the left or right of the ego car?", "answer": LorR},
            {"question": f"Is {kart_name} in front of or behind the ego car?", "answer": ForB},
            {"question": f"Where is {kart_name} relative to the ego car?", "answer": relative_position}
        ])

    qa_pairs.extend([
        {"question": "What kart is the ego car?", "answer": ego_kart_name},
        {"question": "How many karts are there in the scenario?", "answer": str(len(karts))},
        {"question": "What track is this?", "answer": str(track_name)},
        {"question": "How many karts are to the left of the ego car?", "answer": str(left_count)},
        {"question": "How many karts are to the right of the ego car?", "answer": str(len(karts) - 1 - left_count)},
        {"question": "How many karts are in front of the ego car?", "answer": str(front_count)},
        {"question": "How many karts are behind the ego car?", "answer": str(len(karts) - 1 - front_count)}
    ])

    return qa_pairs




def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)

def generate():
    root_data_dir = Path("../data")
    paths = [root_data_dir / "train/", root_data_dir / "valid/"]

    for main_path in paths:
        split_name = main_path.name 

        for file_path in main_path.glob("*_info.json"):
            base_name = file_path.stem.replace("_info", "")

            for image_path in file_path.parent.glob(f"{base_name}_*_im.jpg"):
                stem = image_path.stem
                parts = stem.split('_')
                j = int(parts[-2])

                qa_pairs = generate_qa_pairs(file_path, j)

                for qa in qa_pairs:
                    qa["image_file"] = f"{split_name}/{image_path.name}"
                    qa["question"] = str(qa.get("question", ""))
                    qa["answer"] = str(qa.get("answer", ""))

                new_filename = image_path.stem.replace("_im", "") + "_qa_pairs.json"
                new_file_path = main_path / new_filename

                with open(new_file_path, "w") as f:
                    json.dump(qa_pairs, f, indent=4)

    print("Done.")




def compare_valid_train(valid_balanced_path = "../data/valid_grader/balanced_qa_pairs.json", train_dir = "../data/train/", valid_dir = "../data/valid/"):
   
    with open(valid_balanced_path, "r") as f:
        valid_entries = json.load(f)

    mismatches = []
    matches = []

    for entry in valid_entries:
        image_file = entry.get("image_file")
        question = entry.get("question")
        answer = entry.get("answer")

        if not image_file:
            continue

        
        qa_filename = Path(image_file).stem.replace("_im", "") + "_qa_pairs.json"
        train_qa_path = Path(train_dir) / qa_filename
        valid_qa_path = Path(valid_dir) / qa_filename

        valid_exist = not valid_qa_path.exists()
        train_exist = not valid_qa_path.exists()

        if valid_exist or train_exist :
            mismatches.append({
                "image_file": image_file,
                "reason": "No matching train QA file"
            })
            continue
        
        if not train_exist:
            with open(train_qa_path, "r") as f:
                qas = json.load(f)
        if not valid_exist:
            with open(valid_qa_path, "r") as f:
                qas = json.load(f)

        found = False
    
        for qa in qas:
            if qa.get("answer") == answer and qa.get("question") == question:
                matches.append(entry)
                found = True
                break

        if not found:
            mismatches.append({
                "image_file": image_file,
                "question": question,
                "answer": answer,
                    
            })


    with open("mismatches.json", "w") as f:
        json.dump(mismatches, f, indent=4)

    return mismatches



def check_json_serializable():
    root_data_dir = Path("../data")
    paths = [root_data_dir / "train", root_data_dir / "valid"]

    for main_path in paths:
        for json_file in main_path.glob("*_qa_pairs.json"):
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON in {json_file}: {e}")
                    continue

            for i, item in enumerate(data):
                try:
                    json.dumps(item)
                except (TypeError, OverflowError) as e:
                    print(f"Non-serializable item in {json_file}, index {i}: {item} ({e})")

    print("Check complete.")

check_json_serializable()



    
"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, 
               "generate": generate,
               "check_correct": compare_valid_train,
               "serial": check_json_serializable})


if __name__ == "__main__":
    main()
