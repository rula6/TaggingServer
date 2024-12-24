# Tagging Server
Server that can be queried with images, videos and gifs, and will return associated tags.

Makes use of a `sensitivities.json` file to change sensitivity to certain predicted tags if they're over/under represented vs what you expect or just want to filter some tags from showing up at all.
# Setup
The program uses the [JoyTag](https://github.com/fpgaminer/joytag) ONNX model, so put the `model.onnx` and `top_tags.txt` files in `joytag`.
And do `pip install -r requirements.txt` to install the necessary requirements.