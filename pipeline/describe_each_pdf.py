#!/usr/bin/env python
"""
This file contains all methods and defintions required to generate descriptions for images in a given folder structure.
The folderstructure should look something like this:

basefolder/
├── ansible_news
│   └── whatever.png
└── power_10_news
    └── test.png

The idea is, that in a previous script a folder filled with powerpoint presentations was given and all presentations were converted
into a image per slide. These images should reside inside a sub-folder with the name of the presentation without the extension.

when running this file directly it will output a .desc.txt file next to each image (`test.png` -> `test.png.desc.txt`) with a elaborate description
of the image.
"""

from functools import cache
import os
import logging
from pathlib import Path
from typing import Any

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from config import config


__all__ = [
    "describe_image",
    "IMAGE_DESCRIPTION_PROMPT",
    "run_description_generator",
    "run_description_generator_env",
    "DESCRIPTION_SUFFIX",
    "IMAGE_DESCRIPTION_PROMPT",
]

IMAGE_DESCRIPTION_PROMPT = """
Describe this image.
"""
DEFAULT_MODEL_PATH = config.vision_model_path
DEFAULT_MAX_TOKENS = 1024
DESCRIPTION_SUFFIX = ".desc.txt"

logger = logging.getLogger(__name__)


def _parse_dir(base_dir: Path) -> list[Path]:
    """
    collects a list of image paths from all images located in `base_dir`.

    Args:
        base_dir (Path): The path to start the search from.

    Returns:
        list[Path]: List of paths to all found images
    """
    logger.debug(f"seraching for images in {base_dir}")

    image_extensions = ["png", "jpg", "jpeg", "bmp", "gif", "tiff", "svg", "JPEG"]

    images = []
    for ext in image_extensions:
        for file in base_dir.glob(f"**/*.{ext}"):
            if file.is_file() and os.path.splitext(file.name)[-1] == f".{ext}":
                images.append(file)

    logger.info(f"found {len(images)} images in {base_dir}")

    return images


@cache
def _get_vision_text_model(model_path: str = DEFAULT_MODEL_PATH) -> Any:
    """get the vision model from the module level cache. Load it into memory if this is the first use.

    Args:
        model_path (str, optional): the path to or huggingface repo id of the model. Defaults to DEFAULT_MODEL_PATH.

    Raises:
        ValueError: When the model type pointed to by the model_path is not supported/does not support AutoModelForVision2Seq
                    a ValueError is thrown.

    Returns:
        Any: The loaded Model instance.
    """

    logger.debug(f"loading model {model_path} for first use.")
    #if "pixtral" in model_path.lower():
    #    model = LlavaForConditionalGeneration.from_pretrained(model_path)
    #else:
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype="auto",
            device_map="auto",
        )
    except ValueError as e:
        raise ValueError(
            f'Unsupported Model "{model_path}" provided. Please use model that is either explicitly supported or can use the AutoModelForVision2Seq class.',
        ) from e
    return model


@cache
def _get_vision_processor(model_path: str = DEFAULT_MODEL_PATH) -> Any:
    """get the vision processor from the module level cache. Load it into memory if this is the first use.

    Args:
        model_path (str, optional): the path to or huggingface repo id of the model for which to load the
                                    processor. Defaults to DEFAULT_MODEL_PATH.

    Raises:
        ValueError: When the model type pointed to by the model_path is not supported/does not support AutoProcessor
                    a ValueError is thrown.

    Returns:
        Any: The loaded Model instance.
    """

    logger.debug(f"loading processor for {model_path} for first use.")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        use_fast=True,
        device_map="auto",
    )

    return processor


def describe_image(
    image_path: Path,
    prompt: str = IMAGE_DESCRIPTION_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model_path: str = DEFAULT_MODEL_PATH,
) -> str:
    """uses the Vision-Text-to-Text model at `model_path` to describe the contents of the image.

    Args:
        image_path (Path): path to the image to describe.
        prompt (str, optional): The description prompt to use. Defaults to IMAGE_DESCRIPTION_PROMPT.
        max_tokens (int, optional): max number of tokens in the description. Must be > 0. Defaults to DEFAULT_MAX_TOKENS.
        model_path (str, optional): path or huggingface repo id of the model. Defaults to DEFAULT_MODEL_PATH.

    Returns:
        str: The image description as a string.
    """
    logger.info(f"describing image at {image_path}")

    model = _get_vision_text_model(model_path)
    vision_processor = _get_vision_processor(model_path)
    logger.debug("finished loading models")

    # for processing the vision model requires a conversation template.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = vision_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    logger.debug("generating inputs complete")

    output = model.generate(**inputs, max_new_tokens=max_tokens)
    logger.debug("description generation complete")

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
    ]
    output_text = vision_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # remove chat template before response:
    return output_text[0]


def run_description_generator(
    img_base_dir: Path,
    model_path: str = DEFAULT_MODEL_PATH,
    debug: bool = False,
    prompt: str = IMAGE_DESCRIPTION_PROMPT,
) -> None:
    """runs the :func:`describe_image` function for each image under `img_base_dir` and saves the result at the same
        path with the added suffix ``DESCRIPTION_SUFFIX``

    Args:
        img_base_dir (Path): the directory to search images in (can be located in any sub-level)
        model_path (str, optional): The path to or huggingface id of the model to use for descriptions.
                                    Defaults to DEFAULT_MODEL_PATH.
        debug (bool, optional): whether debugging is active. Will skip images with existing descriptions if False.
                                Defaults to False.
        prompt (str, optional): the system prompt for describing the image. Defaults to IMAGE_DESCRIPTION_PROMPT.
    """
    assert img_base_dir.is_dir(), "Given IMG_DIR does not point to a directory."
    assert prompt, "description prompt may not be empty!"

    for img in _parse_dir(img_base_dir):
        img_desc_path = img.with_suffix(img.suffix + DESCRIPTION_SUFFIX)
        if not debug and img_desc_path.exists() and img_desc_path.stat().st_size != 0:
            # skip already described images

            continue
        logger.info(f"describing image {img} at {img_desc_path}")
        res = describe_image(img, model_path=model_path, prompt=prompt)

        # only write to and thereby create the file after everything was
        # generated and has worked :)
        with open(img_desc_path, "w") as file:
            file.write(res)


def run_description_generator_env() -> None:
    """Loads all available IMG_ variables from the environment and calls :func:`run_description_generator`
       with them.

    Raises:
        ValueError: When the value at `IMG_DIR` does not point to a valid directory.

    The following environment variables can be configured:
        IMG_MODEL (str): path to or huggingface-id of the model to use
        IMG_DIR (str):  path to the base directory und which images should be searched.
        IMG_DEBUG (bool): will enable debugging functionality if present.
    """

    model_path = config.vision_model_path
    logger.info(f"using model path '{model_path}'")

    image_dir = config.output_dir
    logger.info(f"using image directory {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"Path '{image_dir}' given as IMG_DIR is not a directory!")

    if config.debug:
        logger.info("Debugging enabled.")

    run_description_generator(image_dir, config.vision_model_path, config.debug)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    run_description_generator_env()
