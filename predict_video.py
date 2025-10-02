# Prereqs (once): pip install opencv-python pillow torch
# plus the FastVLM repo installed (`pip install -e .`)

MODEL_DIR = "./checkpoints/llava-fastvithd_7b_stage3"
VIDEO_PATH = "/Users/970591/Desktop/VibeCoding/nsflow.mp4"
QUESTION = "What is happening in the video?"

import cv2
import torch
from PIL import Image

from llava.constants import DEFAULT_IM_END_TOKEN
from llava.constants import DEFAULT_IM_START_TOKEN
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import process_images
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model


def load_fastvlm(model_dir: str):
    # device is 'cuda' if NVIDIA GPU available
    #        is 'mps' if Apple GPU is available
    #        is 'cpu' if no GPU is available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    name = get_model_name_from_path(model_dir)

    # tokenizer: A text tokenizer for the underlying language model (e.g. Qwen2, LLaMA, etc.)
    # image_processor: A preprocessing utility for images, typically coming from Hugging Face’s
    #                  transformers library
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_dir, None, name, device=device
    )
    return tokenizer, model, image_processor, device


def prompt_with_image_tokens(question: str, model):
    # If the config has a field mm_use_im_start_end and it’s True,
    # then the model expects special start and end markers around an image token
    # Else, the model does NOT expect special start and end markers around
    # an image token -- The prompt is simpler
    if getattr(model.config, "mm_use_im_start_end", False):
        # Construct the prompt. The prompt is built as:
        # <IM_START> <IMAGE> <IM_END>
        # <question text>
        question = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + question
        )
    else:
        # Construct the prompt. The prompt is built as:
        # <IMAGE>
        # <question text>
        question = DEFAULT_IMAGE_TOKEN + "\n" + question

    # Prepare a conversation template that structures how a question (with an image token
    # already inserted) will be passed into a multimodal model like FastVLM
    # `conv_templates`` is a collection of predefined dialogue formats for different models
    # `qwen_2` refers to the template suitable for Qwen2-based models (which FastVLM often builds on)
    # copy() ensures you don’t overwrite the original template, but work on a fresh copy
    conv = conv_templates["qwen_2"].copy()

    # Add the user’s message (user role)
    conv.append_message(conv.roles[0], question)

    # Add the model’s “reply slot” (model role). This creates an empty slot for the model’s output
    conv.append_message(conv.roles[1], None)

    # Return the final text prompt string that can be fed into the model
    return conv.get_prompt()


@torch.inference_mode()
def ask_frame(
    tokenizer, model, image_processor, device, question: str, pillow_image: Image.Image
):
    # Get prompt for question
    prompt = prompt_with_image_tokens(question, model)

    # Prepare the model input tensor from a text+image prompt so it can be fed into a vision-language model
    input_ids = (
        # tokenizer_image_token(): helper function that uses the tokenizer to turn the prompt string into token IDs
        #     IMAGE_TOKEN_INDEX tells it which special token represents an image. during tokenization,
        #     functions like tokenizer_image_token(...) replace the <image> placeholder in the prompt
        #     with IMAGE_TOKEN_INDEX
        # unsqueeze(0): Adds a new batch dimension at the start. If the tensor shape was [seq_len],
        #               it becomes [1, seq_len]
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device)
    )

    # This takes a raw image (in PIL format), runs it through the image
    # preprocessing pipeline, and prepares it as a tensor for the model
    img = process_images([pillow_image], image_processor, model.config)[0].to(device)

    # If you are on GPU:
    #   casts the tensor to torch.float16 (half precision). Benefit: faster inference, lower memory usage on GPU
    # If you are on CPU:
    #   keep it in torch.float32 (full precision). CPUs generally don’t benefit from float16 and may not support it well
    # unsqueeze(0): Adds a new batch dimension at the front
    img = img.to(torch.float16 if device != "cpu" else torch.float32).unsqueeze(0)

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Generate an answer for the text + image query
    out_ids = model.generate(
        input_ids,
        images=img,
        image_sizes=[pillow_image.size],
        max_new_tokens=128,  # The maximum number of tokens the model is allowed to generate for its answer
        do_sample=True,  # Do stochastic sampling instead of greedy decoding. Makes the model’s answers more diverse
        temperature=0.2,  # Temperature (<1): more deterministic answers. Temperature (>1): more diversee answers
    )

    # Take the raw token IDs output by the model and turn them back into a clean, human-readable string
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def sample_frames(video_path: str, expected_num_of_frames: int = 8):
    # Like callling open() on a file
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"

    # Get total number of frames in video
    actual_num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Get the number of frames the video plays each second
    frames_per_sec = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # The inner list comprehension produces 'expected_num_of_frames' evenly spaced fractional
    # indices between 0 and 'actual_num_of_frames'-1. Each fractional index is rounded to the
    # nearest integer and added to a set. Finally, the set is converted to a sorted list of unique integers
    indicies = sorted(
        {
            int(round(i))
            for i in [
                k * (actual_num_of_frames - 1) / max(1, expected_num_of_frames - 1)
                for k in range(expected_num_of_frames)
            ]
        }
    )

    frames = []
    for idx in indicies:
        # Jump to a specific frame index in the video file
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        # Call read() method on a video capture object
        # ok: True is frame was read successfully. False, if there was an error or we are at the end of the video
        # frame_image: The frame image, a NumPy array containing the pixel data (height × width × channels)
        ok, frame_image = cap.read()
        if ok:
            # Change the color representation of an an image from BGR format to RGB
            # OpenCV loads images in BGR order by default (Blue, Green, Red). Most other
            # libraries (like matplotlib, PIL, or deep learning frameworks) expect RGB order (Red, Green, Blue)
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

            # Save a frame and its timestamp into a tuple
            # Tuple's first element: converts the NumPy array into a PIL (Python Image Library) Image object
            # Tuple's second element: converts the frame index into time in seconds, rounded to 2 decimal places
            frames.append(
                (Image.fromarray(frame_image), round(idx / frames_per_sec, 2))
            )

    # release the video capture resource (like close() method on an open file)
    cap.release()

    return frames


if __name__ == "__main__":
    tokenizer, model, image_processor, device = load_fastvlm(MODEL_DIR)
    for pillow_image, timestamp in sample_frames(VIDEO_PATH, expected_num_of_frames=8):
        answer = ask_frame(
            tokenizer, model, image_processor, device, QUESTION, pillow_image
        )
        print(f"[t={timestamp:6.2f}s] {answer}")
