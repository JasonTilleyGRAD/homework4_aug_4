#Ai used to help write data validation functions

from pathlib import Path
import fire
from matplotlib import pyplot as plt
import json

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
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

    captions =[]
    for kart in karts:
        if kart.get("kart_name") == ego_kart_name:
            continue
        coord = kart.get("center")
        x, y = float(coord[0]), float(coord[1])

        offset_x = int(x - ego_kart_center[0])
        offset_y = int(y - ego_kart_center[1])

    
        if abs(offset_x) > abs(offset_y):
            if offset_x < 0:
                position = "right of"
            else:
                position = "left of"
        else:
            if offset_y > 0:
                position = "in front of"
            else:
                position = "behind"
        captions.append({"caption":  str(kart["kart_name"]) + " is " + str(position) +" the ego car." })

    

    
    captions.append({"caption": str(ego_kart_name)+  " is the ego car."})
    captions.append({"caption": "There are " +  str(len(karts)) + " karts in the scene."})
    captions.append({"caption": "The track is " + str(track_name) +"."})
        
    return captions


    


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

def generate():
    root_data_dir = Path("data/")
    paths = [root_data_dir / "train/", root_data_dir / "valid/"]

    for main_path in paths:
        split_name = main_path.name

        for file_path in main_path.glob("*_info.json"):
            base_name = file_path.stem.replace("_info", "")

            for image_path in file_path.parent.glob(f"{base_name}_*_im.jpg"):
                
                stem = image_path.stem
                parts = stem.split('_')
                j = int(parts[-2])

                captions = generate_caption(file_path, j)

                for cap in captions:
                    cap["image_file"] = f"{split_name}/{image_path.name}"
                    

                new_filename = image_path.stem.replace("_im", "") + "_captions.json"
                new_file_path = main_path / new_filename
               
                with open(new_file_path, "w") as f:
                    json.dump(captions, f, indent=4)

    print("Done.")


def compare_valid_train(valid_balanced_path="data/valid_grader/all_mc_qas.json", train_dir="data/train",valid_dir="data/valid"):

    with open(valid_balanced_path, "r") as f:
        valid_entries = json.load(f) 

    mismatches = []

    for entry in valid_entries:
        image_file = entry.get("image_file")
        if not image_file:
            continue

        caption_file = Path(image_file).stem.replace("_im", "") + "_captions.json"
        train_qa_path = Path(train_dir) / caption_file
        valid_qa_path = Path(valid_dir) / caption_file

        train_exists = train_qa_path.exists()
        valid_exists = valid_qa_path.exists()

        if not train_exists and not valid_exists:
            mismatches.append({
                "image_file": image_file,
                "reason": "No matching train or valid QA file"
            })
            continue

        caption_path = train_qa_path if train_exists else valid_qa_path
        with open(caption_path, "r") as f:
            generated_entries = json.load(f)  

        correct_index = entry.get("correct_index")
        correct_caption = entry.get("candidates", [])[correct_index]

        found = False
        

        
        for gen_entry in generated_entries:
            if correct_caption == gen_entry.get("caption", []):
                found = True
                break

        if not found:
            mismatches.append({
                "image_file": image_file,
                "correct caption": correct_caption 
            })


    print(f"Total: {len(mismatches)}/{len(valid_entries)}")
    return mismatches



"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, 
               "generate": generate,
               "checker": compare_valid_train})


if __name__ == "__main__":
    main()
