import chromadb
import numpy as np
import matplotlib.pyplot as plt
import umap
import random

chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_collection(name="zalgorithm")

results = collection.get(include=["embeddings", "metadatas", "documents"])

embeddings = np.array(results["embeddings"])
metadatas = results["metadatas"]

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")

embedding_2d = reducer.fit_transform(embeddings)

plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=140, alpha=0.75)

for i, metadata in enumerate(metadatas):
    print(f"{i}: {metadata['page_title']}: {metadata['section_heading']}")
    label = str(i)
    x = embedding_2d[i, 0]
    y = embedding_2d[i, 1]
    plt.text(x, y, label, ha="center", va="center", color="white", fontsize=6)

plt.show()
