from pathlib import Path

import fire
from matplotlib import pyplot as plt
import json

from .generate_qa import draw_detections, extract_frame_info,extract_kart_objects,extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    with open(info_path, 'r') as f:
        file = json.load(f)
    track_name = extract_track_info(file)

    karts = extract_kart_objects(file,view_index,img_width,img_height)

    ego_kart_name = None
    for kart in karts:
        if kart.get("is_center_kart", True):
            ego_kart_name = kart["kart_name"]
            ego_kart_center = kart["center"]
            break

    captions =[]
    for kart in karts:
        coord = kart["center"]
        new_coors = (int(coord[0] - ego_kart_center[0])),int(coord[1] - ego_kart_center[1])


        if abs(new_coors[0]) > abs(new_coors[1]):
            if new_coors[0] > 0:
                position = "right"
            else:
                position = "left"
        else:
            if new_coors[1] > 0:
                position = "above"
            else:
                position = "below"
        captions.append({"caption": f"{kart["kart_name"]} is {position} of the ego car."})

    

    
    captions.append({"caption": f"{ego_kart_name} is the ego car."})
    captions.append({"caption": f"There are {len(karts)} karts in the scenario."})
    captions.append({"caption": f"The track is {track_name}."})
        
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


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()
