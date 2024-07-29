# Simple RAG
This is a toy project with two implementations of a RAG setup. It is not meant for production use at all, but can be useful for testing ideas.

A sample dataset is provided, but any type of document can be used.

## Implementation 1
A RAG setup coded from scratch taking the BERT model/tokenizer and faiss as the index. The implementation comprises two files:
- [index_documents.py](index_documents.py) This script processes the three sample dataset files by first chunking each document into chunks of 512 tokens. Each chunk contains some metadata that helps linking context between chunks. The embeddings are then generated for each chunk and the chunks saved into a chunks directory. The index is generated through faiss.
- [answer.py](answer.py) This script loads the index previously created and then encodes a query that will be searched on the index. The index returns the top k more relevant documents for the query and then a context is provided to a very simple language model that answers the query with the context.

This is a very simple, from scratch, implementation. It does not work very well, and in some cases it does not even fetch the right document for a very simple query. This is probably because the tokenizers, encoders and models are quite old, but it's also possible that there is a bug in the implementation.

```
python index_documents.py
```

```
python answer.py --query "What was the Operating Income for AAPL on 2022?"
```

## Implementation 2
This RAG setup leverages the use of amazing already built libraries, like ragatouille and mixedbread. The implementation also comprises two files:
- [index_mixedbread.py](index_mixedbread.py) A RAG index is built using the mixedbread model and the ragatouille wrapper. The index is saved on a .ragatouille directory.
- [answer_mixedbread.py](answer_mixedbread.py) This script loads the previously created index and answers a user provided query. The result shows the most relevant documents fetched. A Language model could be added on top to answer the user query for a more rounded implementation.

This is a very simple implementation as well. The use of third party libraries elevates the implementation, and the results are quite decent. This is not meant to be used in production, but a production ready version would have a similar skeleton.

```
python index_mixedbread.py
```

```
python answer_mixedbread.py --query "What was the Operating Income for AAPL on 2022?"
```

### Note
Since two implementations are done the requeriments.txt provided are quite bloated. A virtual environment is heavily encouraged.
