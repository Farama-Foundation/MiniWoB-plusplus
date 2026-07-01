"""Screenshot utilities."""
import json
from io import BytesIO
from typing import cast

import numpy as np
from PIL import Image, ImageCms, ImageDraw
from selenium.webdriver import Chrome as ChromeDriver


def get_screenshot(
    driver: ChromeDriver,
    true_width: int,
    true_height: int,
    crop_width: int,
    crop_height: int,
) -> Image.Image:
    """Return a scaled and cropped screenshot taken by the Selenium instance.

    Args:
        driver: Chrome WebDriver.
        true_width: Width of the screenshot in the correct resolution.
        true_height: Height of the screenshot in the correct resolution.
        crop_width: Width to crop the image to.
        crop_height: Height to crop the image to.

    Returns:
        A PIL Image object with width crop_width and height crop_height.
    """
    png_data = driver.get_screenshot_as_png()
    pil_image = Image.open(BytesIO(png_data)).convert("RGB")
    pil_image = pil_image.crop(
        (
            0,
            0,
            crop_width * pil_image.size[0] // true_width,
            crop_height * pil_image.size[1] // true_height,
        )
    )
    pil_image = pil_image.resize((crop_width, crop_height))
    icc_profile = pil_image.info.get("icc_profile")
    if icc_profile:
        orig_icc = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
        srgb_icc = ImageCms.createProfile("sRGB")
        pil_image = cast(
            Image.Image, ImageCms.profileToProfile(pil_image, orig_icc, srgb_icc)
        )
    return pil_image


def pil_to_numpy_array(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to a numpy array.

    Args:
        pil_image: PIL Image.

    Returns:
        A numpy array of shape (height, width, 3)
        where 3 is the number of channels (RGB).
    """
    return np.array(pil_image).astype(np.uint8)


def create_gif(path_prefix: str):
    """Create and save an animated gif based on the dumped screenshots.

    The event file is read from <path_prefix>.json, while the images are
    loaded from <path_prefix>-<step>.png

    Args:
        path_prefix: The path prefix, such as
            'data/experiments/123_unnamed/traces/test/2000-img/2000-3'
            (control step 2000; episode 3)
    """
    # Read the event file
    with open(path_prefix + ".json") as fin:
        events = json.load(fin)
    # Read the image files
    images = []
    for i, event in enumerate(events):
        img = Image.open(f"{path_prefix}-{i}.png").convert("RGBA")
        images.append(img)
        # Highlight the element
        if "element" in event:
            elt = event["element"]
            highlight = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(highlight)
            x0 = elt["left"]
            x1 = x0 + elt["width"]
            y0 = elt["top"]
            y1 = y0 + elt["height"]
            draw.rectangle(
                (x0, y0, x1, y1), fill=(255, 0, 0, 128), outline=(0, 0, 255, 255)
            )
            del draw
            images.append(Image.alpha_composite(img, highlight))
    # Save the image file
    durations = [250] * len(images)
    durations[-1] = 1000
    images[0].save(
        path_prefix + ".gif",
        append_images=images[1:],
        save_all=True,
        loop=0,
        duration=durations,
    )
