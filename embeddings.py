# from langchain_community.embeddings import OllamaEmbeddings

# import boto3
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

from langchain_nomic.embeddings import NomicEmbeddings

def get_embeddings():
    # embeddings = OllamaEmbeddings(model='llama2')

    # bedrock_client = boto3.client(
    #     service_name='bedrock_runtime', 
    #     region_name='us-east-1'
    #     )
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name='default',
    #     region_name='us-east-1',
    #     )
    
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

    return embeddings
