import os
import cv2
import pickle
import numpy as np
import face_recognition
import os
from dotenv import load_dotenv
from typing import List, Tuple

# Load the environment variables from the .env file
load_dotenv()

known_encodings_path: str = os.getenv("PEOPLE_TO_IDENTIFY_ENCODINGS")
dataset_encodings_path: str = os.getenv("DATASET_ENCODINGS")


def saveEncodings(encs: List[np.ndarray], names: List[str], fname: str = "encodings.pickle") -> None:
    """
    Save encodings in a pickle file to be used in the future.

    Parameters:
    encs (List of np arrays): List of face encodings.
    names (List of strings): List of names for each face encoding.
    fname (String, optional): Name/Location for the pickle file. Default is "encodings.pickle".
    """
    data = []
    d = [{"name": nm, "encoding": enc} for (nm, enc) in zip(names, encs)]
    data.extend(d)

    encodingsFile: str = fname

    # Dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    f = open(encodingsFile, "wb")
    f.write(pickle.dumps(data))
    f.close()


def readEncodingsPickle(fname: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Read a Pickle file.

    Parameters:
    fname (String): Name of the pickle file (full location).

    Returns:
    encodings (list of np arrays): List of all saved encodings.
    names (List of Strings): List of all saved names.
    """
    data = pickle.loads(open(fname, "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]
    names = [d["name"] for d in data]
    return encodings, names


def createEncodings(image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Create face encodings for a given image and also return face locations in the given image.

    Parameters:
    image (cv2 mat): Image you want to detect faces from.

    Returns:
    known_encodings (list of np array): List of face encodings in a given image.
    face_locations (list of tuples): List of tuples for face locations in a given image.
    """
    # Find face locations for all faces in an image
    face_locations = face_recognition.face_locations(image)

    # Create encodings for all faces in an image
    known_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    return known_encodings, face_locations


def compareFaceEncodings(unknown_encoding: np.ndarray, known_encodings: List[np.ndarray], known_names: List[str]) -> Tuple[bool, str, float]:
    """
    Compare face encodings to check if 2 faces are the same or not.

    Parameters:
    unknown_encoding (np array): Face encoding of unknown people.
    known_encodings (np array): Face encodings of known people.
    known_names (list of strings): Names of known people.

    Returns:
    acceptBool (Bool): Face matched or not.
    duplicateName (String): Name of matched face.
    distance (Float): Distance between 2 faces.
    """
    duplicateName = ""
    distance = 0.0
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)
    distance = face_distances[best_match_index]
    if matches[best_match_index]:
        acceptBool = True
        duplicateName = known_names[best_match_index]
    else:
        acceptBool = False
        duplicateName = ""
    return acceptBool, duplicateName, distance


def saveImageToDirectory(image: np.ndarray, name: str, imageName: str) -> None:
    """
    Save images to a directory.

    Parameters:
    image (cv2 mat): Image you want to save.
    name (String): Directory where you want the image to be saved.
    imageName (String): Name of the image.
    """
    path = "./output/" + name
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    cv2.imwrite(path + "/" + imageName, image)


def processKnownPeopleImages(path: str = "./people_to_identify/", saveLocation: str = known_encodings_path) -> None:
    """
    Process images of known people and create face encodings to compare in the future.
    Each image should have just 1 face in it.

    Parameters:
    path (STRING, optional): Path for known people dataset. Default is "./people_to_identify/".
                             It should be noted that each image in this dataset should contain only 1 face.
    saveLocation (STRING, optional): Path for storing encodings for known people dataset.
                                     Default is "./known_encodings.pickle" in the current directory.
    """
    known_encodings: List[np.ndarray] = []
    known_names: List[str] = []
    for img in os.listdir(path):
        imgPath = path + img

        # Read image
        image = cv2.imread(imgPath)
        name = img.rsplit('.')[0]
        # Resize
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

        # Get locations and encodings
        encs, locs = createEncodings(image)

        known_encodings.append(encs[0])
        known_names.append(name)

        for loc in locs:
            top, right, bottom, left = loc

        # Show Image
        cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    saveEncodings(known_encodings, known_names, saveLocation)


def processDatasetImages(path: str = "./dataset/", saveLocation: str = dataset_encodings_path) -> None:
    """
    Process images in the dataset from where you want to separate images.
    It separates the images into directories of known people, groups, and any unknown people images.

    Parameters:
    path (STRING, optional): Path for the dataset. Default is "./dataset/".
                             It should be noted that each image in this dataset should contain only 1 face.
    saveLocation (STRING, optional): Path for storing encodings for the dataset.
                                     Default is "./dataset_encodings.pickle" in the current directory.
    """
    # Read pickle file for known people to compare faces from
    people_encodings, names = readEncodingsPickle(known_encodings_path)

    for img in os.listdir(path):
        imgPath = path + img

        # Read image
        image = cv2.imread(imgPath)
        orig = image.copy()

        # Resize
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

        # Get locations and encodings
        encs, locs = createEncodings(image)

        # Save image to a group image folder if more than one face is in the image
        if len(locs) > 1:
            saveImageToDirectory(orig, "Group", img)

        # Processing image for each face
        i = 0
        knownFlag = 0
        for loc in locs:
            top, right, bottom, left = loc
            unknown_encoding = encs[i]
            i += 1
            acceptBool, duplicateName, distance = compareFaceEncodings(unknown_encoding, people_encodings, names)
            if acceptBool:
                saveImageToDirectory(orig, duplicateName, img)
                knownFlag = 1
        if knownFlag == 1:
            print("Match Found")
        else:
            saveImageToDirectory(orig, "Unknown", img)

        # Show Image
        cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()


def main() -> None:
    """
    Main Function.
    """
    datasetPath = "./dataset/"
    peoplePath = "./people_to_identify/"
    processKnownPeopleImages(path=peoplePath)
    processDatasetImages(path=datasetPath)
    print("Completed")


if __name__ == "__main__":
    main()
