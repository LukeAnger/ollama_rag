# from langchain_community.embeddings import OllamaEmbeddings

# import boto3
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

# from langchain_nomic.embeddings import NomicEmbeddings

import voyageai
voyageai.api_key = "pa-kQPbUw14f_f_j7C5ShvSa34N9Q_X2wjVIv41PFN6rPk"

def get_embeddings(docs):
    # embeddings = OllamaEmbeddings(model='llama2')
    # bedrock_client = boto3.client(
    #     service_name='bedrock_runtime', 
    #     region_name='us-east-1'
    #     )

    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name='default',
    #     region_name='us-east-1',
    #     )
    # embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    print("Getting embeddings: ", docs)
    vo = voyageai.Client()
    embeddings = vo.embed(
        docs,
        model="voyage-large-2-instruct",
        input_type="document"
    ).embeddings

    return embeddings
