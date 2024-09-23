import os
from rembg import remove

base_dir = "saves"
output_dir = "saves_rmbg"

os.makedirs(output_dir, exist_ok=True)
for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)

    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith(".png"):
                input_path = os.path.join(subdir_path, filename)
                output_path = os.path.join(f"{output_dir}\{subdir}", filename)
                with open(input_path, "rb") as input_file:
                    img = input_file.read()
                output_img = remove(img)
                with open(output_path, "wb") as output_file:
                    output_file.write(output_img)

                print(f"Processed {filename} from {subdir}")

print("Background removal completed.")
