import io
import base64

from PIL import Image, ImageDraw


def serialize(image: Image, format: str = "JPEG") -> str:
    """Converts PIL image to base64 string."""

    buffer = io.BytesIO()
    image.save(buffer, format=format)
    byte_string = buffer.getvalue()
    base64_string = base64.b64encode(byte_string).decode()
    return base64_string


def deserialize(base64_string: str) -> Image:
    """Converts base64 string to PIL image."""
    decoded_string = base64.b64decode(base64_string)
    buffer = io.BytesIO(decoded_string)
    return Image.open(buffer)
