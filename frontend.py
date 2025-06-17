
import logging
import gradio as gr
import chromadb
from chromadb.utils import embedding_functions
from config import config

logger = logging.getLogger(__name__)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.sentence_model_path
)


def fetch_slides(prompt: str) -> list[str]:
    """
    Fetches the top four slides best matching the prompt.

    Args:
        prompt (str): The prompt given by the user
    Returns:
        list[str]: A list of length <=4 with paths to the images best matching the prompt.
    """
    logger.debug(f"processing prompt '{prompt}'")

    # Initialize Chroma client and collection
    chroma_client = chromadb.PersistentClient(
        path=str(config.database_dir)
    )  # Change path accordingly
    collection = chroma_client.get_collection(
        name="all_files", embedding_function=sentence_transformer_ef
    )

    results = collection.query(
        query_texts=[prompt], n_results=config.n_results, include=["documents"]
    )

    logger.debug(f"results: {results}")

    # result_two = results["ids"][0][0][1:]

    ids = [id.replace("../", "./") for id in results["ids"][0]]
    # Generate the image using the model
    return ids


with gr.Blocks() as demo:
    gr.Markdown("# Searching through Slides on IBM Power")

    question = gr.Textbox(
        label="Enter a description of the slide that you are looking for"
    )
    submit_button = gr.Button("Lookup Slides")
    images = gr.Gallery(
        label="Generated Images",
        allow_preview=True,
        scale=0.2,
        columns=4,
        object_fit="contain",
    )

    action = [fetch_slides, question, images]

    question.submit(*action)
    submit_button.click(*action)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=config.server_port,
        allowed_paths=[
        ],
    )