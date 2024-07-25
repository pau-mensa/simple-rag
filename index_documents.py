from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import numpy as np
import faiss
import torch
import os

print("Loading checkpoints...")
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
print("Loaded!")

def chunk_text(text, metadata, chunk_size=512, overlap=50):
    tokens = context_tokenizer.tokenize(text)
    chunks = []
    count = 0
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(
            f"[METADATA: Chunk number {count} | {' | '.join([f'{k}: {v}' for k, v in metadata.items()])}] \n" + context_tokenizer.convert_tokens_to_string(chunk)
        )
        count += 1
    return chunks

def encode_document(document, metadata):
    chunks = chunk_text(document, metadata)
    chunk_embeddings = []

    for chunk in chunks:
        inputs = context_tokenizer(chunk, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = context_encoder(**inputs).pooler_output
        chunk_embeddings.append(embeddings.numpy())
    
    return chunk_embeddings, chunks

if __name__=='__main__':
    doc_names = ['dataset/filtered_AAPL.csv', 'dataset/filtered_NVDA.csv', 'dataset/filtered_MSFT.csv']
    documents = []
    for doc in doc_names:
        with open(doc, 'r') as f:
            documents.append(f.read())

    context_embeddings = []
    chunked_docs = []
    print("Tokenizing...")
    os.makedirs('chunks/', exist_ok=True)
    for doc_idx, doc in enumerate(documents):
        embedding, chunks = encode_document(doc, {
            'Document ID': doc_idx,
            'Company Name': ''.join([char for char in doc_names[doc_idx] if char.isupper()])
        })
        context_embeddings.extend(embedding)
        chunked_docs.extend([f"{doc_names[doc_idx].split('.csv')[0].split('/')[-1]}_{c_id}.txt" for c_id in range(len(embedding))])

        for c_id, c in enumerate(chunks):
            with open(f"chunks/{doc_names[doc_idx].split('.csv')[0].split('/')[-1]}_{c_id}.txt", 'w') as f:
                f.write(c)
    print("Tokenized!")
    context_embeddings = np.vstack(context_embeddings)

    index = faiss.IndexFlatL2(context_embeddings.shape[1])
    index.add(context_embeddings)

    faiss.write_index(index, 'index.faiss')

    with open('documents.txt', 'w') as f:
        for doc in chunked_docs:
            f.write("%s\n" % doc)

    print("DONE")
