import gradio as gr
import numpy as np
from PIL import Image
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import cv2 as cv2
import insightface
from insightface.app import FaceAnalysis

load_dotenv()

mongodb_connection_string = os.environ.get('MONGODB_CONNECTION_STRING')

# MongoDB setup
mongo_client = MongoClient(mongodb_connection_string)
db = mongo_client["pov_image_similarity"]
collection = db["photos_embedded"]
image_directory = "photos"

# Initialize face analysis once
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

def vectorize_image(image):
    faces = app.get(image)
    print(f"# of faces detected - {len(faces)}")
    
    if len(faces) == 0:
        return None
    
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

def process_image(image):
    if image is None:
        return None, "Please upload an image"
    
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Vectorize the uploaded image
    vector = vectorize_image(opencv_image)
    
    if vector is None:
        return None, "No faces detected in the image. Please upload a photo with exactly 1 face."
    
    # Search for similar images
    similar_images = search_similar_images(vector)
    
    if not similar_images:
        return None, "No similar images found in the database."
    
    # Prepare results for display
    result_images = []
    result_text = []
    
    for doc in similar_images:
        try:
            img_path = doc['image_path']
            score = doc['score']
            face_no = doc['faceNo']
            
            # Load and display the similar image
            similar_img = Image.open(img_path)
            result_images.append(similar_img)
            result_text.append(f"Vector Search Score: {score:.4f}\nFace No: {face_no}")
            
            print(f"{img_path} - Face {face_no}: {score}")
        except Exception as e:
            print(f"Error loading image {doc['image_path']}: {e}")
    
    if result_images:
        return result_images, result_text
    else:
        return None, "Error loading similar images from database."

def create_interface():
    with gr.Blocks(title="Similar ID Search", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Similar ID Search")
        gr.Markdown("Upload a photo with exactly 1 face to find similar faces in the database.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Photo (ONLY 1 FACE!)",
                    height=400
                )
                search_btn = gr.Button("Search Similar IDs", variant="primary")
            
            with gr.Column(scale=1):
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Upload an image to start searching..."
                )
                
                with gr.Column():
                    result_gallery = gr.Gallery(
                        label="Similar IDs Found",
                        show_label=True,
                        elem_id="gallery",
                        columns=1,
                        rows=3,
                        height="auto",
                        allow_preview=True
                    )
        
        def handle_search(image):
            if image is None:
                return gr.update(), "Please upload an image first."
            
            try:
                result_images, result_info = process_image(image)
                
                if result_images is None:
                    return gr.update(value=[]), result_info
                
                # Format images with captions for gallery
                gallery_data = []
                for img, info in zip(result_images, result_info):
                    gallery_data.append((img, info))
                
                return gr.update(value=gallery_data), f"Vector Search Completed."
                
            except Exception as e:
                return gr.update(value=[]), f"Error: {str(e)}"
        
        search_btn.click(
            handle_search,
            inputs=[image_input],
            outputs=[result_gallery, status_text]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)