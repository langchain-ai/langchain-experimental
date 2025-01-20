import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from langchain_experimental.graph_transformers import LLMGraphTransformer

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-2024-08-06")
llm_transformer = LLMGraphTransformer(llm=llm)


def test_save_load_graph_document() -> None:
    """Test to check if the graph document is saved and loaded correctly"""
    text = """
      Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
      She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
      Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
      She was, in 1906, the first woman to become a professor at the University of Paris.
    """  # noqa: E501
    documents = [Document(page_content=text)]
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    intermediate_file_name = "graph_document.pkl"
    llm_transformer.save_graph_documents(graph_documents, intermediate_file_name)

    loaded_graph_documents = llm_transformer.load_graph_documents(
        intermediate_file_name
    )
    # deleting the file after testing
    os.remove(intermediate_file_name)
    # checking all the both graph documents are same or not
    assert graph_documents == loaded_graph_documents
