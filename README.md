# Face matching with ID

```bash
brew install uv

uv venv --python 3.12
source .venv/bin/activate

uv sync
```

Copy the environment variable template and fill in necessary information

```bash
cp .env.sample .env
```

Place IDs in the folder `photos`

```bash
mkdir -p photos
```

Load the images, embed and store to MongoDB Atlas

```bash
python3 embed_images.py
```

Create the following Vector Search index in the collection

```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 512,
    "similarity": "euclidean"
  }]
}
```

Run the image search UI

```bash
streamlit run image_search.py
```
