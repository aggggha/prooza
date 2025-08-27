import faiss
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load FAISS index
index = faiss.read_index("faiss/index.faiss")
vectors = index.reconstruct_n(0, index.ntotal)

# Save vectors to TSV
np.savetxt("output_index/vectors.tsv", vectors, delimiter="\t")

# Load metadata (optional)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db = FAISS.load_local("faiss", embedding_model, allow_dangerous_deserialization=True)

metadata = [doc.page_content for doc in db.docstore._dict.values()]
pd.Series(metadata).to_csv("output_index/metadata.tsv", index=False, header=False)