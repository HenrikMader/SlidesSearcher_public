#!/usr/bin/env python
from enum import Enum, auto
from functools import cache
import logging
import torch
import os
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
    SentenceTransformerEmbeddingFunction,
)
from chromadb import Collection
from tqdm import tqdm

from config import config

#try:
    # running from external/main.py
#    from describe_each_pdf import describe_image
#except ModuleNotFoundError:
    # running file standalone
#    from describe_each_pdf import describe_image

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)


class CollectionStatus(Enum):
    COLLECTION_CREATED = auto()
    COLLECTION_EXISTS = auto()
    COLLECTION_CREATION_FAILED = auto()


@cache
def _get_sentence_transformer(
    model_name: str = "all-mpnet-base-v2",
) -> SentenceTransformerEmbeddingFunction:
    """gets the SentenceTransformer used for chromadb. Used for lazy loading.

    Args:
        model_name (str, optional): The name of the model to use. Defaults to "all-mpnet-base-v2".

    Returns:
        SentenceTransformerEmbeddingFunction: The embedding function for chroma db.
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def ensure_collection(
    client: chromadb.ClientAPI, collection_name: str
) -> tuple[CollectionStatus, Optional[Collection]]:
    """Get a collection from the db with the given name. If no collection with that name exists
    a new one is created.

    Args:
        client (chromadb.ClientAPI): the chromadb client connection
        collection_name (str): the collection name to get

    Returns:
        tuple[CollectionStatus, Optional[Collection]]: Status of the operation, collection if it existed or could be created.
                                                        None in case of an error.
    """
    try:
        # Check if the collection already exists
        collection = client.get_collection(
            name=collection_name, embedding_function=_get_sentence_transformer()
        )
        logger.debug(f"Collection '{collection_name}' already exists.")
        return CollectionStatus.COLLECTION_EXISTS, collection

    except Exception:
        # If it doesn't exist, create a new collection
        try:
            collection = client.create_collection(
                name=collection_name, embedding_function=_get_sentence_transformer()
            )
            logger.debug(f"Collection '{collection_name}' created successfully.")
            return CollectionStatus.COLLECTION_CREATED, collection
        except Exception as e:
            logger.exception(f"Failed to create collection '{collection_name}': {e}")
            return CollectionStatus.COLLECTION_CREATION_FAILED, None


def insert_files_into_db(db_dir: str):
    """Opens os creates a ChromDB at the given dir. Then inserts all pngs from
    the ../finalDestination/ directory into the database using a desription from descriptions.py.

    All files will be inserted into a single collection called "all_files".

    Args:
        db_dir (str): The directory for the db to store its files.
    """
    print("db_dir")
    print(db_dir)
    db_dir = str(db_dir)
    chroma_client = chromadb.PersistentClient(path=db_dir)
    collection_status, collection = ensure_collection(chroma_client, "all_files")

    for folder in tqdm(os.listdir(config.output_dir), desc="Inserting Presentations"):
        for file in tqdm(
            os.listdir(config.output_dir / folder), desc="Describing Slides"
        ):
            print(file)
            if (".desc.txt" not in str(file)):
                img_path = config.output_dir / folder / file

                #output = describe_image(
                #    Path(img_path), model_path=str(config.vision_model_path)
                #)

                output = open(str(img_path) + ".desc.txt")
                output = output.read()
                collection.add(
                    documents=[output],
                    metadatas=[{"image_path": str(img_path), "presentation": str(folder)}],
                    ids=[str(img_path)],
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if config.debug else logging.INFO)
    insert_files_into_db(config.database_dir)
