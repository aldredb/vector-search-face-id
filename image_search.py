import streamlit as st
import numpy as np
from PIL import Image
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import cv2 as cv2
from glob import glob
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

load_dotenv()

mongodb_connection_string = os.environ.get('MONGODB_CONNECTION_STRING')

# MongoDB setup
mongo_client = MongoClient(mongodb_connection_string)
db = mongo_client["pov_image_similarity"]
collection = db["photos_embedded"]
image_directory = "photos"

def vectorize_image(image):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(image)

    print(f"# of faces detected - {len(faces)}")
    # ONLY RETURN THE FIRST FACE
    embedding = faces[0].normed_embedding

    return embedding

def search_similar_images(image_vector, threshold=0.8):
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": image_vector.tolist(),
                "path": "embedding",
                "numCandidates": 30,
                "index": "default",
                "limit": 3
            }
        },
        {
            "$project": {
                "image_path": 1,
                "faceNo": 1,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    ]
    result = list(collection.aggregate(pipeline))
    return result


def main():
    st.title("Similar ID Search")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    # User input in the left column
    with col1:
        uploaded_file = st.file_uploader("ONLY UPLOAD PHOTO WITH 1 FACE!!!")
        
        if uploaded_file is not None:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Photo.", use_container_width=True)
            
            # Vectorize the uploaded image
            vector = vectorize_image(opencv_image)
            similar_images = search_similar_images(vector)
            
            print(len(similar_images))
    
    # Display results in the right column
    with col2:
        if 'similar_images' in locals():
            st.header("Similar IDs")
            for doc in similar_images:
                print(f"{doc['image_path']} - Face {str(doc['faceNo'])}: {str(doc['score'])}")
                st.image(doc['image_path'], use_container_width=True)
                # Display score with larger text using markdown
                st.markdown(f"<h4 style='text-align: center;'>Vector Search Score: {doc['score']:.4f}</h4>", unsafe_allow_html=True)
                st.divider()  # Add divider between results

if __name__ == "__main__":
    main()