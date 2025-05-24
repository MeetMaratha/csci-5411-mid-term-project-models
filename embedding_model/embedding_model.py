import numpy as np
from flask import Flask, Response, json, make_response, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
model: SentenceTransformer = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
)


@app.route("/get_embeddings", methods=["POST"])
def getEmbeddings() -> Response:

    request_data = request.json

    chunks: str = request_data["chunks"]

    embeddings_list = []
    for chunk in chunks:
        embeddings = model.encode(chunk)
        embeddings_list.append(embeddings.astype(np.float64).tolist())
    return make_response(json.dumps({"data": embeddings_list}), 200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
