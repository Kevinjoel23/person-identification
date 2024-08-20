**dataset** - The directory where all your images are kept.

**people_to_identify** - This folder should contain the images of the people you want to identify in the dataset.

**known_encodings** - This folder is used to save the encodings of the people you want to identify.

**dataset_encodings** - This folder is used to save the encodings of all the images in the dataset, which will be compared with the known encodings.

**Quickstart**

To get started, follow these steps:

1. Install the following libraries in the virtual environment of your choice:
```
    - dlib
    - opencv
    - face_recognition
    - pickle
```

2. Place the required images in the `dataset` and `people_to_identify` folders.

3. Activate the environment and run the following command:

```bash
python image_segmentation.py
```
