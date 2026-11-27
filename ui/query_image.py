"""Validate an in-session upload and prepare an image query."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps, UnidentifiedImageError


def prepare_image(upload):
    if upload.size > 5 * 1024 * 1024:
        raise ValueError("Maximum image size is 5 MB.")

    try:
        with Image.open(BytesIO(upload.getvalue())) as source:
            if source.width * source.height > 20_000_000:
                raise ValueError("Image resolution is too large.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((1024, 1024))
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValueError("Choose a valid JPG or PNG image.") from error


def encode_query_image(engine, image):
    with TemporaryDirectory() as directory:
        path = Path(directory) / "query.jpg"
        path.write_bytes(image)
        return engine.relevance.encode_image(path)


def prepare_query(engine, image, session, cache_key):
    key = (engine.relevance.model_version, image)
    cached = session.get(cache_key)
    if cached is None or cached[0] != key:
        cached = (key, encode_query_image(engine, image))
        session[cache_key] = cached
    return engine.relevance.evaluate_embedding(cached[1])
