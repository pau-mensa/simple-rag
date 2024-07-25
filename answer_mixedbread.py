from ragatouille import RAGPretrainedModel
import pickle
import argparse
import os


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Ask a query to a RAG index')

    parser.add_argument('--query', required=True, type=str,
                        help='Query to ask the RAG')
    parser.add_argument('--index_path', default=None, type=str,
                        help='Path of the directory of the index. If not passed then the index path will be inferred from the index_name.pkl')

    args = parser.parse_args()

    if args.index_path is None:
        print("Index path not provided, trying index_name.pkl")
        with open('index_name.pkl', 'rb') as handle:
            idx_path = pickle.load(handle)
    else:
        idx_path = args.index_path

    assert os.path.isdir(idx_path), "index_path must be a directory."

    RAG = RAGPretrainedModel.from_index(idx_path)
    query = args.query
    results = RAG.search(query, k=2)
    print(results)
