from mixedbread_ai.client import MixedbreadAI
from ragatouille import RAGPretrainedModel
import pickle

MXBAI = MixedbreadAI(api_key="emb_b01f94c0cae8641cfce160997d2d1d8322068d390ed1a3f0")
RAG = RAGPretrainedModel.from_pretrained("mixedbread-ai/mxbai-colbert-v1")

def encode_documents(documents):
    """
    Function to get the embeddings of a series of documents. 
    We don't need it in this implementation since the RAG already encodes the documents.
    """
    res = MXBAI.embeddings(
        model='mixedbread-ai/mxbai-embed-large-v1',
        input=documents,
        normalized=True,
        encoding_format='float',
        truncation_strategy='end'
    )
    
    return [x.embedding for x in res.data]

def get_ticker(doc_name):
    return ''.join([char for char in doc_name if char.isupper()])

if __name__=='__main__':
    doc_names = ['dataset/filtered_AAPL.csv', 'dataset/filtered_NVDA.csv', 'dataset/filtered_MSFT.csv']
    documents = []
    for doc in doc_names:
        with open(doc, 'r') as f:
            documents.append(f.read())

    document_metadatas = [
        {"company": k} for k in [get_ticker(d) for d in doc_names]
    ]

    idx_path = RAG.index(
        collection=documents, 
        document_ids=[str(x) for x in range(len(doc_names))],
        document_metadatas=document_metadatas,
        index_name="results"
    )
    print(f'Index saved at {idx_path}')

    with open('index_name.pkl', 'wb') as handle:
        pickle.dump(idx_path, handle, protocol=pickle.HIGHEST_PROTOCOL)
