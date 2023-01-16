import io

import torch

from segmentation import get_segmentor, get_segments
from starlette.responses import Response
import uvicorn
from fastapi import FastAPI, File

model = get_segmentor()


app = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


@app.post("/segmentation")
def get_segmentation_map(file: bytes = File(...)):
    """Get segmentation maps from image file"""
    segmented_image = get_segments(model, file)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)