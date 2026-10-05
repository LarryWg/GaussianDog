"""Port the isolated BITE source to the installed rendering API."""

from pathlib import Path


def patch_source(source):
    path = Path(source) / "src/smal_pytorch/renderer/differentiable_renderer.py"
    original = path.read_text()
    updated = original.replace(
        "elif pytorch3d.__version__ == '0.6.1':", "elif pytorch3d.__version__ != '0.2.5':"
    )
    updated = updated.replace("Materials, Textures,", "Materials,")
    updated = updated.replace("Textures(verts_rgb=tex)", "TexturesVertex(verts_features=tex)")
    if updated != original:
        path.write_text(updated)
    image_path = Path(source) / "src/stacked_hourglass/utils/pilutil.py"
    original_image = image_path.read_text()
    updated_image = original_image.replace("Image.isImageType(im)", "isinstance(im, Image.Image)")
    if updated_image != original_image:
        image_path.write_text(updated_image)
    return path
