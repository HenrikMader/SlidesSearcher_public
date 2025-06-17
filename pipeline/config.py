from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SlideSearcherSettings(BaseSettings):
    """
    All configurations for the slidesearche application. Automatically parsed from the environment.
    """

    model_config = SettingsConfigDict(
        env_prefix="SLIDES_",
        case_sensitive=False,
        env_parse_none_str=True,
    )

    vision_model_path: Path = Field(
        alias="SLIDES_VISION_MODEL", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )

    sentence_model_path: str = Field(
        alias="SLIDES_SENTENCE_MODEL", default="all-mpnet-base-v2"
    )

    pptx_dir: Path = Field(alias="SLIDES_PPTX_DIR", default="./Files/PPTX_DIR/")
    pdf_dir: Path = Field(alias="SLIDES_PDF_DIR", default="./Files/PDF_DIR/")
    output_dir: Path = Field(alias="SLIDES_IMG_DIR", default="./Files/IMG_DIR/")
    database_dir: Path = Field(alias="SLIDES_DB_DIR", default="db/")
    n_results: int = 4

    server_port: int = Field(alias="SLIDES_PORT", default=7680)

    debug: bool = False


config = SlideSearcherSettings()
