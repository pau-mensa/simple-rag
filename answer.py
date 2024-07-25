import os

# In some local environments not setting this to True crashes the program.
# Warning: According to the crash error, setting this to True can lead to wrong generations.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, T5Tokenizer, T5ForConditionalGeneration
import faiss
import argparse

print("Loading checkpoints...")
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
print("Loaded!")

def retrieve(query, k=1):
    inputs = question_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    question_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    distances, indices = index.search(question_embedding, k)
    return [documents[i] for i in indices[0] if i != -1]

def generate_response(query, retrieved):
    context = ''.join(retrieved)
    input_text = f'query: {query}. context: {context}'
    inputs = generator_tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    output = generator_model.generate(**inputs, max_new_tokens=1000)
    return generator_tokenizer.decode(output[0], skip_special_tokens=True)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Ask a query to a RAG index')

    parser.add_argument('--query', required=True, type=str,
                        help='Query to ask the RAG')
    parser.add_argument('--index_path', default=None, type=str,
                        help='Path of the directory of the index. If not passed then the index path will be inferred from the index_name.pkl')

    args = parser.parse_args()

    if args.index_path is None:
        idx_path = 'index.faiss'
    else:
        idx_path = args.index_path

    assert os.path.isfile(idx_path), "Index_path must be a file"

    index = faiss.read_index(idx_path)
    with open('documents.txt', 'r') as f:
        documents = [line.strip() for line in f.readlines()]
    
    generator_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    generator_model = T5ForConditionalGeneration.from_pretrained('t5-base')

    query = args.query
    retrieved_docs = retrieve(query)
    print(retrieved_docs)
    retrieved = ''
    for doc in retrieved_docs:
        with open(f'chunks/{doc}', 'r') as f:
            retrieved = retrieved.join(f.readlines())
    response = generate_response(query, retrieved)
    print(response)
