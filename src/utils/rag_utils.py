def load_documents(filenames, data_root):
    """
    Read all documents that are passed in filenames
    """
    documents = []
    for file in filenames:
        loader = TextLoader(data_root+file)
        document = loader.load()
        for d in document:
            d.metadata['source'] = '/'.join(file.split('/')[-2:])
        documents += document
    return documents


def split_documents(documents, chunk_size=512, overlap=100):
    """
    Split the documents into chunks for further processing
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    documents = text_splitter.split_documents(documents)
    return documents


def get_hf_embeddings_model():
    """
    Get the embedding model to create our vector_store.
    The embedding model to use is fine tuned on a medical dataset.
    
    @software{balachandran2024medembed,
    author = {Balachandran, Abhinand},
    title = {MedEmbed: Medical-Focused Embedding Models},
    year = {2024},
    url = {https://github.com/abhinand5/MedEmbed}
    }
    """
    model_name ="abhinand/MedEmbed-small-v0.1"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return model