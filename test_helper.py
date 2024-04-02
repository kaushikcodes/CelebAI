from PIL import Image
import os

test_dir = 'test_kag'

for filename in os.listdir(test_dir):
    if filename.lower().endswith('.jpg'):
        path = os.path.join(test_dir, filename)
        try:
            # Open and immediately save the image to normalize it
            image = Image.open(path)
            print(type(image))
            image.save(path, 'JPEG')
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
