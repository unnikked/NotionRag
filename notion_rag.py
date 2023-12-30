"""
This script demonstrates how to preprocess and index documents using the LangChain library. 
It uses a Notion database as the source of documents, splits the documents into smaller 
chunks, preprocesses the metadata, and then builds a vector index using the Chroma vector store.

Summary:
1. The script loads documents from a Notion database using the NotionDBLoader class.
2. It splits the documents into smaller chunks using the RecursiveCharacterTextSplitter class.
3. The metadata of each split is preprocessed to handle any null values or complex data types.
4. The split documents are then indexed and stored in the Chroma vector store using the OpenAIEmbeddings for encoding the text.
5. The index is built and stored in the specified directory.

By following these steps, the script enables efficient similarity search and retrieval of documents based on their textual content.
"""

import dotenv
import os

dotenv.load_dotenv()

def import_pipeline():

    # These three lines swap the stdlib sqlite3 lib with the pysqlite3 package
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

    from langchain.document_loaders import NotionDBLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    import json

    def preprocess_metadata(metadata):
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ''
            elif isinstance(value, list):
                metadata[key] = ', '.join(value)
            elif isinstance(value, dict):
                metadata[key] = json.dumps(value)
        return metadata

    loader = NotionDBLoader(
        integration_token=os.environ['NOTION_API_KEY'],
        database_id=os.environ['NOTION_DATABASE_ID'],
        request_timeout_sec=30  # Optional, defaults to 10
    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = preprocess_metadata(split.metadata)

    print('Building the index on Chroma...')
    Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory="./chroma")

if __name__ == '__main__':
    import_pipeline()