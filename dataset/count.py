import os 

print("Counting the number of images in the dataset...")
image_list = os.listdir("sft_img")
print(f"Number of images in the dataset: {len(image_list)}")
print("Counting the number of captions in the dataset...")