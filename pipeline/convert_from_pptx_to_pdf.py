#!/usr/bin/env python
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys

from pdf2image import convert_from_path
from tqdm import tqdm

from config import config

logger = logging.getLogger(__name__)


def convert_file(input_file: Path, output_dir: Path) -> None:
    """converts a given pptx file to pdf and places the pdf file into the output directory.

    Args:
        input_file (str): file to convert (must be a readable by powerpoint/libreoffice)
        output_dir (str): directory to place the resulting pdf file in.

    Raises:
        OSError: When the OS could not be detected or no conversion mechanism is implemented for
                 this OS.
        ValueError: When the input_file is not a powerpoint file or output_dir exists but is not a directory.
    """
    current_os = platform.system()

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(
            f"output dir '{output_dir}' passed to convert_file is not directory!"
        )
    elif not output_dir.exists():
        os.mkdir(output_dir)

    if current_os == "Windows":
        # NOTE: requires PowerPoint to be installed
        # import fails on non Windows hosts.
        from pptxtopdf import convert

        convert(input_file, output_dir)
    elif current_os == "Linux" or current_os == "Darwin":
        # NOTE: requires libreoffice to be installed.
        cmd = "soffice"  # assumes libreoffice to be in path.
        if current_os == "Darwin":
            cmd = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

        subprocess.run(
            [
                cmd,
                "--headless",
                "--convert-to",
                "pdf",
                f"{input_file}",
                "--outdir",
                f"{output_dir}",
            ],
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    else:
        raise OSError(
            f"unsupported operating system '{current_os}' in use. Cannot convert PPTX files to pdfs. please do so manually!"
        )


def convert_files_in_dir(
    pptx_dir: Path, output_dir: Path, delete_intermediates: bool = True
) -> None:
    """Converts all powerpoint files found in the directory into folders containing pngs of every slide.

    Args:
        pptx_dir (Path): directory containing the powerpoint files to process
        output_dir (Path): directory under which each powerpoint should get its sub-dir
        delete_intermediates (bool, optional): Whether to delete the intermidiate pdf file. Defaults to True.
    """

    logger.debug(f"Starting to convert files from {pptx_dir}")
    for file in tqdm(os.listdir(pptx_dir), desc="Processing Powerpoints"):
        logger.debug(f"converting file {file}.")
        name, ext = os.path.splitext(file)

        if ext.lower() not in (".ppt", ".pptx"):
            # file is not a supported powerpoint presentation.
            # skipping.
            logger.debug(f"file {file} is not a supported presentation.")
            continue

        input_file = pptx_dir / file

        if not os.path.isdir(output_dir / name):
            logger.debug("Output dir does not exist, creating.")
            os.mkdir(output_dir / name)

        pdf_file_name = name + ".pdf"
        if not (output_dir / name / pdf_file_name).exists():
            logger.debug(f"creating PDF for {file}")
            convert_file(input_file, output_dir / name)
        else:
            logger.debug(f"file {file} already was converted to a PDF.")

        images_from_path = convert_from_path(
            output_dir / name / pdf_file_name,
            output_folder=output_dir / name,
            fmt="png",
            size=(800, None),
        )

        if delete_intermediates:
            os.remove(output_dir / name / pdf_file_name)


if __name__ == "__main__":
    logging.basicConfig()
    convert_files_in_dir(config.pptx_dir, config.output_dir, delete_intermediates=True)
