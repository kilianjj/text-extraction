
import os
import shutil

# prep the data for use by PyTorch

# follow the expected files structure for ImageFolder
# do one set for digits for a simple example

data_dir = '/Users/kilianjj/Documents/Code/letter_recognition/data/archive/Img'
digits_dir = '/Users/kilianjj/Documents/Code/letter_recognition/data/digits/dataset'
train = "train"
test = "val"

def make_test_val_dirs():
    for i in range(0, 10):
        os.makedirs(os.path.join(digits_dir, train, str(i)), exist_ok=True)
        os.makedirs(os.path.join(digits_dir, test, str(i)), exist_ok=True)

def populate_classes():
    for file in sorted(os.listdir(data_dir)):
        try:
            # Extract digit from filename
            digit = int(file.split('-')[0][-3:]) - 1
            count = int(file.split('-')[1].split('.')[0])
            if digit > 9:
                continue
            # Decide if file goes to train or test set
            dest_dir = test if count < 11 else train
            shutil.copy(os.path.join(data_dir, file), os.path.join(digits_dir, dest_dir, str(digit), file))
        except ValueError as e:
            print(f"Skipping file {file} due to error: {e}")


if not os.path.exists(data_dir):
    print(f"Error: Data directory '{data_dir}' not found.")
    exit(1)

make_test_val_dirs()
populate_classes()
