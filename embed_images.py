from PIL import Image
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import cv2 as cv2
from glob import glob
import insightface
from insightface.app import FaceAnalysis
from insightface.data  import get_image as ins_get_image

load_dotenv()

mongodb_connection_string = os.environ.get('MONGODB_CONNECTION_STRING')

print(mongodb_connection_string)
# MongoDB setup
mongo_client = MongoClient(mongodb_connection_string)
db = mongo_client["pov_image_similarity"]
collection = db["photos_embedded"]
image_directory = "photos"

def vectorize_and_store_images(image_directory):

    img_paths = glob(f'./{image_directory}/*')

    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    

    for img_path in img_paths:
       print(f"Processing {img_path}")
       img = cv2.imread(img_path)
       if img is None: continue
       faces = app.get(img)

       # If there are no faces, then skip     
       if len(faces) < 1:
           print("There are no faces detected in this photo")
           continue
        
       # There can be many faces detected in 1 photo. We generate 1 embedding for each photo
       print(f"# of faces detected - {len(faces)}")

       for index, face in enumerate(faces):
            print(f"Getting embedding for {img_path} - Face {index}")
            embedding = face.normed_embedding
            # print(face.normed_embedding)

            document = {
                "image_path": img_path,
                "faceNo": index,
                "embedding": embedding.tolist(),
            }

            # Insert into MongoDB - 1 document per embedding/face
            # This means that there can be more than 1 document per photo
            collection.insert_one(document)
            print(f"Inserted embedding for {img_path} - Face {index} into MongoDB.")

def main():
    vectorize_and_store_images(image_directory)

if __name__ == "__main__":
    main()
