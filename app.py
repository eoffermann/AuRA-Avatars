from pathlib import Path
import gradio as gr
import inference
import json
import random
import os
import pprint
import tempfile
from moviepy import VideoFileClip
from PIL import Image

from typing import List

SETTINGS_FILE = "settings.json"

def extract_first_frame_as_png(mp4_path: str) -> str:
    """
    Extracts the first frame of an MP4 file as a PNG image and saves it to a temporary location.

    Args:
        mp4_path (str): The path to the input MP4 file.

    Returns:
        str: The path to the saved PNG image.
    """
    if mp4_path is None:
        return mp4_path
    
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(f"The file at path {mp4_path} does not exist.")

    if not os.path.splitext(mp4_path)[-1]==".mp4":
        # If it's not an mp4, just return the original path. Let the calling routine decide what to
        # do if it can't read the file.
        return mp4_path
    
    # Load the video file
    with VideoFileClip(mp4_path) as video:
        # Get the first frame as an image
        first_frame = video.get_frame(0)

    # Create a temporary file to save the PNG
    temp_dir = tempfile.gettempdir()
    temp_png_path = os.path.join(temp_dir, "first_frame.png")

    # Save the first frame as a PNG
    image = Image.fromarray(first_frame)
    image.save(temp_png_path, format="PNG")
    return temp_png_path

def load_settings():
    """Load settings from `settings.json` if it exists."""
    if Path(SETTINGS_FILE).is_file():
        with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def save_settings(settings):
    """Save settings to `settings.json`."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=4)

def load_options():
    """Load options from `options.json`"""
    fh=open('options.json','r')
    data=json.load(fh)
    fh.close()
    for key in data:
        data[key]=sort_custom(data[key])
    data['object_a']=data['objects']
    data['object_b']=data['objects']
    return data

def build_prompt(age="middle aged", race="generic", gender="female", hair_length="short", hair_texture="curly", 
                 hair_color="dark", object_a="lamp", object_b="plant", clothing_tone="dark", clothing_type="suit", 
                 shirt_color="white", shirt_type="dress shirt", accessory="tie", broadcast_type="news", emotion="serious", camera="The camera is stationary and locked off"):
    # Derive pronouns from gender. Ideally, we'd have something more nuanced to support 
    # more ambiguously gendered / androgynous avatars but that's poorly represented in
    # the datasets and wildly hit-or-miss for generation
    if (gender=="man"):
        pronoun="he"
        poss_pronoun="his"
    else:
        pronoun="she"
        poss_pronoun="her"
    
    hair_prompt=f"A {age} {race} {gender} with {hair_length} {hair_color} hair"
    if (hair_length=="shaved") & (hair_texture=="bald"):
        hair_prompt=f"A {age} {race} {gender} with a {hair_length} head"

    gen_prompt=f"{hair_prompt} looks directly into the camera. The man is {emotion}. The background is out of focus and contains {object_a} and {object_b}. {pronoun}'s wearing a {clothing_tone} {clothing_type} with a {shirt_color} {shirt_type} and {accessory}. {pronoun} blinks and nods {poss_pronoun} head and looks intently into the camera. {camera} framing {poss_pronoun} head and shoulders. High quality professional lighting. The scene appears to be from a {broadcast_type} broadcast."
    return gen_prompt

def custom_sort_key(item: str) -> str:
    """
    Sorting key function that skips the prefixes 'a ', 'an ', and 'the ' if present.
    """
    # Define a list of prefixes to ignore
    prefixes = ["a ", "an ", "the "]
    
    # Check if the string starts with any of the prefixes (case-insensitive)
    for prefix in prefixes:
        if item.lower().startswith(prefix):
            return item[len(prefix):].lower()  # Remove the prefix for sorting key
    
    # If no prefix matches, return the whole string
    return item.lower()

def sort_custom(items: List[str]) -> List[str]:
    """
    Sorts a list of strings alphabetically, but ignores the prefixes 'a ', 'an ', and 'the '.
    """
    return sorted(items, key=custom_sort_key)

def get_video_gallery() -> list[dict]:
    """
    Recursively search for MP4 files in a given directory and return a list of dictionaries
    with video metadata.

    Args:
        directory (str): The directory path to search for videos.

    Returns:
        list[dict]: A list of dictionaries, each containing 'name' and 'data' for videos.
    """
    directory='outputs'
    videos = []
    here=os.path.dirname(__file__)
    # Walk through the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith("_hd.mp4"):  # Check if the file is an HD MP4
                relative_path = os.path.relpath(os.path.join(root, file), start=here)
                videos.append([relative_path, os.path.basename(file)])
    return videos

def display_file(image_path):
    """
    Displays the uploaded file.
    """
    if image_path is None:
        return gr.update(visible=False), gr.update(visible=False)  # Hide both outputs
    
    file_path = image_path.name
    if file_path.endswith(('.jpg', '.png')):
        # Show the image, hide the video
        return gr.update(value=file_path, visible=True), gr.update(visible=False)
    elif file_path.endswith('.mp4'):
        # Show the video, hide the image
        return gr.update(visible=False), gr.update(value=file_path, visible=True)
    else:
        # Hide both if unsupported type
        return gr.update(visible=False), gr.update(visible=False)

def gradio_interface(
    seed,
    num_inference_steps,
    num_images_per_prompt,
    height,
    width,
    num_runs,
    num_frames,
    frame_rate,
    extend_clip,
    restart_first_frame,
    upscale_clip,
    age,
    race,
    gender,
    hair_length,
    hair_texture,
    hair_color,
    object_a,
    object_b,
    clothing_tone,
    clothing_type,
    shirt_color,
    shirt_type,
    accessory,
    broadcast_type,
    emotion,
    camera,
    input_image_path=None
) -> str:
    """
    Gradio interface wrapper for the video generation pipeline.
    """
    print(f"gradio interface iip: {input_image_path}")
    input_image_path=extract_first_frame_as_png(input_image_path)
    print(f"after iip: {input_image_path}")
    # Prepare arguments as a dictionary
    args = {
        "ckpt_dir": "PATH",
        "input_image_path": input_image_path,
        "output_path": None,
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "num_runs": int(num_runs),
        "num_images_per_prompt": int(num_images_per_prompt),
        "guidance_scale": 3.0,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "frame_rate": int(frame_rate),
        "bfloat16": False,
        "prompt": build_prompt(age=age, race=race, gender=gender, hair_length=hair_length, 
                               hair_texture=hair_texture, hair_color=hair_color, object_a=object_a,
                               object_b=object_b, clothing_tone=clothing_tone, clothing_type=clothing_type,
                               shirt_color=shirt_color, shirt_type=shirt_type, accessory=accessory, broadcast_type=broadcast_type,
                               emotion=emotion, camera=camera),
        "negative_prompt": "cropped, letterboxed, text, logo, graphics, worst quality, deformed, distorted, inconsistent motion, blurry, jittery, distorted",
        "extend_clip": extend_clip,
        "restart_first_frame": restart_first_frame,
        "upscale_clip": upscale_clip,
        "override_filename": f"{emotion}_{age}_{race}_{gender}_{hair_length}_{hair_texture}_{hair_color}"
    }

    # Save current settings
    save_settings(args)

    # Run the pipeline
    try:
        inference.run_pipeline(**args)
        return f"Avatar Generation complete."
    except Exception as e:
        return f"Error: {str(e)}"

def get_value_or_random(key, value):
    if (value=="Random"):
        r=random.choice(options[key])
        print(f"options ({options[key]}): {r}")
        return r
    else:
        print(f"value: ({value})")
        return value

def bulk_gradio_interface(
    seed,
    num_inference_steps,
    num_images_per_prompt,
    height,
    width,
    num_runs,
    num_frames,
    frame_rate,
    extend_clip,
    upscale_clip,
    age,
    race,
    gender,
    hair_length,
    hair_texture,
    hair_color,
    object_a,
    object_b,
    clothing_tone,
    clothing_type,
    shirt_color,
    shirt_type,
    accessory,
    broadcast_type,
    emotion,
    camera,
    input_image_path=None
) -> str:
    """
    Bulk Gradio interface wrapper for the video generation pipeline.
    """
    input_image_path=extract_first_frame_as_png(input_image_path)
    errors=0
    for run in range(0, num_runs):
        bulk_age=get_value_or_random("age", age)
        bulk_race=get_value_or_random("race", race)
        bulk_gender=get_value_or_random("gender", gender)
        bulk_hair_length=get_value_or_random("hair_length", hair_length)
        bulk_hair_texture=get_value_or_random("hair_texture", hair_texture)
        bulk_hair_color=get_value_or_random("hair_color", hair_color)
        bulk_object_a=get_value_or_random("object_a", object_a)
        bulk_object_b=get_value_or_random("object_b", object_b)
        bulk_clothing_tone=get_value_or_random("clothing_tone", clothing_tone)
        bulk_clothing_type=get_value_or_random("clothing_type", clothing_type)
        bulk_shirt_color=get_value_or_random("shirt_color", shirt_color)
        bulk_shirt_type=get_value_or_random("shirt_type", shirt_type)
        bulk_accessory=get_value_or_random("accessory", accessory)
        bulk_broadcast_type=get_value_or_random("broadcast_type", broadcast_type)
        bulk_emotion=get_value_or_random("emotion", emotion)
        bulk_camera=get_value_or_random("camera", camera)
        # Prepare arguments as a dictionary
        args = {
            "ckpt_dir": "PATH",
            "input_image_path": input_image_path,
            "output_path": None,
            "seed": int(seed),
            "num_inference_steps": int(num_inference_steps),
            "num_runs": 1,
            "num_images_per_prompt": int(num_images_per_prompt),
            "guidance_scale": 3.0,
            "height": int(height),
            "width": int(width),
            "num_frames": int(num_frames),
            "frame_rate": int(frame_rate),
            "bfloat16": False,
            "prompt": build_prompt(age=bulk_age, race=bulk_race, gender=bulk_gender, hair_length=bulk_hair_length, 
                                    hair_texture=bulk_hair_texture, hair_color=bulk_hair_color, object_a=bulk_object_a,
                                    object_b=bulk_object_b, clothing_tone=bulk_clothing_tone, clothing_type=bulk_clothing_type,
                                    shirt_color=bulk_shirt_color, shirt_type=bulk_shirt_type, accessory=bulk_accessory, broadcast_type=bulk_broadcast_type,
                                    emotion=bulk_emotion, camera=bulk_camera),
            "negative_prompt": "cropped, letterboxed, lower third, title, text, logo, graphics, worst quality, deformed, distorted, inconsistent motion, blurry, jittery, distorted",
            "extend_clip": extend_clip,
            "restart_first_frame": False,
            "upscale_clip": upscale_clip,
            "override_filename": f"{bulk_emotion}_{bulk_age}_{bulk_race}_{bulk_gender}_{bulk_hair_length}_{bulk_hair_texture}_{bulk_hair_color}"
        }

        # Save current settings
        pprint.pprint(args, indent=4)
    
        # Run the pipeline
        try:
            inference.run_pipeline(**args)
        except Exception as e:
            errors+=1
        print("="*80)

    if (errors>0):
        return "Avatar generation complete with errors - check log for details"
    else:
        return "Avatar generation completed successfully"

    
# Load default settings
default_settings = load_settings()
options = load_options()

# Gradio interface setup
footer="""
footer {
    visibility: hidden;
}"""
css = f"""
{footer}
"""
theme = gr.themes.Ocean(
    primary_hue="violet",
    secondary_hue="rose",
    neutral_hue="slate",
)

with gr.Blocks(title="AuRA Avatar Generator", css=css, theme=theme) as app:
    gr.Markdown("# AuRA Avatar Generator\n*from Big Blue Ceiling*")
    with gr.Tabs():
        with gr.Tab(label="Generate an Avatar Set"):
            gr.Markdown("## Select avatar parameters and generate a selection of avatar animations for one avatar")
            with gr.Row():
                with gr.Column(scale=1):
                    age=gr.Dropdown(choices=options['age'], label="Age", value=random.choice(options['age']))
                    race=gr.Dropdown(choices=options['race'], label="Race", value=random.choice(options['race']))
                    gender=gr.Dropdown(choices=options['gender'], label="Gender", value=random.choice(options['gender']))
                    hair_length=gr.Dropdown(choices=options['hair_length'], label="Hair length", value=random.choice(options['hair_length']))
                    hair_texture=gr.Dropdown(choices=options['hair_texture'], label="Hair texture", value=random.choice(options['hair_texture']))
                    hair_color=gr.Dropdown(choices=options['hair_color'], label="Hair color", value=random.choice(options['hair_color']))
                    object_a=gr.Dropdown(choices=options['objects'], label="BG Object A", value=random.choice(options['objects']))
                    object_b=gr.Dropdown(choices=options['objects'], label="BG Object B", value=random.choice(options['objects']))
                with gr.Column(scale=1):
                    clothing_tone=gr.Dropdown(choices=options['clothing_tone'], label="Clothing tone", value=random.choice(options['clothing_tone']))
                    clothing_type=gr.Dropdown(choices=options['clothing_type'], label="Clothing type", value=random.choice(options['clothing_type']))
                    shirt_color=gr.Dropdown(choices=options['shirt_color'], label="Shirt color", value=random.choice(options['shirt_color']))
                    shirt_type=gr.Dropdown(choices=options['shirt_type'], label="Shirt type", value=random.choice(options['shirt_type']))
                    accessory=gr.Dropdown(choices=options['accessory'], label="Accessory", value=random.choice(options['accessory']))
                    broadcast_type=gr.Dropdown(choices=options['broadcast_type'], label="Broadcast type")
                    emotion=gr.Dropdown(choices=options['emotion'], label="Emotion", value=random.choice(options['emotion']))
                    camera=gr.Dropdown(choices=options['camera'], label="Camera movement", value=random.choice(options['camera']))
                with gr.Column(scale=3):
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed", 
                            value=default_settings.get("seed", -1), 
                            precision=0
                        )
                        num_inference_steps = gr.Slider(
                            label="Number of Inference Steps", 
                            minimum=1, 
                            maximum=100, 
                            value=default_settings.get("num_inference_steps", 40), 
                            step=1,
                            visible=False
                        )
                        num_runs = gr.Slider(
                            label="Number of runs to generate", 
                            minimum=1, 
                            maximum=100, 
                            value=default_settings.get("num_runs", 1), 
                            step=1
                        )
                        num_images_per_prompt = gr.Slider(
                            label="Number of avatar videos per run", 
                            minimum=1, 
                            maximum=100, 
                            value=default_settings.get("num_images_per_prompt", 1), 
                            step=1
                        )

                    with gr.Row():
                        height = gr.Slider(
                            label="Height", 
                            minimum=64, 
                            maximum=1080, 
                            value=default_settings.get("height", 416), 
                            step=32, 
                            visible=False
                        )
                        width = gr.Slider(
                            label="Width", 
                            minimum=64, 
                            maximum=1920, 
                            value=default_settings.get("width", 768), 
                            step=32, 
                            visible=False
                        )
                        num_frames = gr.Slider(
                            label="Number of Frames", 
                            minimum=1, 
                            maximum=300, 
                            value=default_settings.get("num_frames", 121),
                            visible=False
                        )
                        frame_rate = gr.Slider(
                            label="Frame Rate", 
                            minimum=1, 
                            maximum=60, 
                            value=default_settings.get("frame_rate", 24)
                        )

                    with gr.Row():
                        extendClip = gr.Checkbox(
                            label="Create extended clip series", 
                            value=default_settings.get("extend_clip", True)
                        )
                        restart_first_frame = gr.Checkbox(
                            label="Restart extended clip with first frame", 
                            value=default_settings.get("restart_first_frame", False)
                        )
                        upscale_clip = gr.Checkbox(
                            label="Upscale", 
                            value=default_settings.get("upscale_clip", False)
                        )

                    with gr.Row():
                        input_image_path = gr.File(label="Input Image Path (Optional)", file_types=[".jpg", ".png", ".mp4"])
                    with gr.Row():
                        output_image = gr.Image(label="Image Preview", visible=False)
                        output_video = gr.Video(label="Video Preview", visible=False)

                    input_image_path.change(
                        display_file,
                        inputs=[input_image_path],
                        outputs=[output_image, output_video]
                    )

                    generate_button = gr.Button("Generate Avatars")
            output_message = gr.Textbox(label="Status", value="Ready")

            generate_button.click(
                gradio_interface,
                inputs=[
                    seed,
                    num_inference_steps,
                    num_images_per_prompt,
                    height,
                    width,
                    num_runs,
                    num_frames,
                    frame_rate,
                    extendClip,
                    restart_first_frame,
                    upscale_clip,
                    age,
                    race,
                    gender,
                    hair_length,
                    hair_texture,
                    hair_color,
                    object_a,
                    object_b,
                    clothing_tone,
                    clothing_type,
                    shirt_color,
                    shirt_type,
                    accessory,
                    broadcast_type,
                    emotion,
                    camera,
                    input_image_path
                ],
                outputs=output_message,
            )
        with gr.Tab(label="Batch Generate Avatars"):
            gr.Markdown("## Batch generation of avatars that share certain characteristics - randomizing other attributes")
            with gr.Row():
                with gr.Column(scale=1):
                    bulk_age=gr.Dropdown(choices=['Random']+options['age'], label="Age", value="Random")
                    bulk_race=gr.Dropdown(choices=['Random']+options['race'], label="Race", value="Random")
                    bulk_gender=gr.Dropdown(choices=['Random']+options['gender'], label="Gender", value="Random")
                    bulk_hair_length=gr.Dropdown(choices=['Random']+options['hair_length'], label="Hair length", value="Random")
                    bulk_hair_texture=gr.Dropdown(choices=['Random']+options['hair_texture'], label="Hair texture", value="Random")
                    bulk_hair_color=gr.Dropdown(choices=['Random']+options['hair_color'], label="Hair color", value="Random")
                    bulk_object_a=gr.Dropdown(choices=['Random']+options['objects'], label="BG Object A", value="Random")
                    bulk_object_b=gr.Dropdown(choices=['Random']+options['objects'], label="BG Object B", value="Random")
                with gr.Column(scale=1):
                    bulk_clothing_tone=gr.Dropdown(choices=['Random']+options['clothing_tone'], label="Clothing tone", value="Random")
                    bulk_clothing_type=gr.Dropdown(choices=['Random']+options['clothing_type'], label="Clothing type", value="Random")
                    bulk_shirt_color=gr.Dropdown(choices=['Random']+options['shirt_color'], label="Shirt color", value="Random")
                    bulk_shirt_type=gr.Dropdown(choices=['Random']+options['shirt_type'], label="Shirt type", value="Random")
                    bulk_accessory=gr.Dropdown(choices=['Random']+options['accessory'], label="Accessory", value="Random")
                    bulk_broadcast_type=gr.Dropdown(choices=['Random']+options['broadcast_type'], label="Broadcast type")
                    bulk_emotion=gr.Dropdown(choices=['Random']+options['emotion'], label="Emotion", value="Random")
                    bulk_camera=gr.Dropdown(choices=['Random']+options['camera'], label="Camera movement", value="Random")

                with gr.Column(scale=3):
                    with gr.Row():
                        bulk_seed = gr.Number(
                            label="Seed", 
                            value=default_settings.get("seed", -1), 
                            precision=0
                        )
                        bulk_num_inference_steps = gr.Slider(
                            label="Number of Inference Steps", 
                            minimum=1, 
                            maximum=100, 
                            value=default_settings.get("num_inference_steps", 40), 
                            step=1,
                            visible=False
                        )
                        bulk_num_runs = gr.Slider(
                            label="Number of runs to generate", 
                            minimum=1, 
                            maximum=100, 
                            value=default_settings.get("num_runs", 1), 
                            step=1
                        )
                        bulk_num_images_per_prompt = gr.Slider(
                            label="Number of avatar videos per run", 
                            minimum=1, 
                            maximum=100, 
                            value=default_settings.get("num_images_per_prompt", 1), 
                            step=1
                        )

                    with gr.Row():
                        bulk_height = gr.Slider(
                            label="Height", 
                            minimum=64, 
                            maximum=1080, 
                            value=default_settings.get("height", 416), 
                            step=32, 
                            visible=False
                        )
                        bulk_width = gr.Slider(
                            label="Width", 
                            minimum=64, 
                            maximum=1920, 
                            value=default_settings.get("width", 768), 
                            step=32, 
                            visible=False
                        )
                        bulk_num_frames = gr.Slider(
                            label="Number of Frames", 
                            minimum=1, 
                            maximum=300, 
                            value=default_settings.get("num_frames", 121),
                            visible=False
                        )
                        bulk_frame_rate = gr.Slider(
                            label="Frame Rate", 
                            minimum=1, 
                            maximum=60, 
                            value=default_settings.get("frame_rate", 24)
                        )

                    with gr.Row():
                        bulk_extendClip = gr.Checkbox(
                            label="Create extended clip series", 
                            value=default_settings.get("extend_clip", True)
                        )
                        bulk_upscale_clip = gr.Checkbox(
                            label="Upscale", 
                            value=default_settings.get("upscale_clip", False)
                        )

                    with gr.Row():
                        bulk_input_image_path = gr.File(label="Input Image Path (Optional)", file_types=[".jpg", ".png", ".mp4"])
                    with gr.Row():
                        bulk_output_image = gr.Image(label="Image Preview", visible=False)
                        bulk_output_video = gr.Video(label="Video Preview", visible=False)

                    bulk_input_image_path.change(
                        display_file,
                        inputs=[bulk_input_image_path],
                        outputs=[bulk_output_image, bulk_output_video]
                    )


                    bulk_generate_button = gr.Button("Generate Avatars")
            bulk_output_message = gr.Textbox(label="Status", value="Ready")

            bulk_generate_button.click(
                bulk_gradio_interface,
                inputs=[
                    bulk_seed,
                    bulk_num_inference_steps,
                    bulk_num_images_per_prompt,
                    bulk_height,
                    bulk_width,
                    bulk_num_runs,
                    bulk_num_frames,
                    bulk_frame_rate,
                    bulk_extendClip,
                    bulk_upscale_clip,
                    bulk_age,
                    bulk_race,
                    bulk_gender,
                    bulk_hair_length,
                    bulk_hair_texture,
                    bulk_hair_color,
                    bulk_object_a,
                    bulk_object_b,
                    bulk_clothing_tone,
                    bulk_clothing_type,
                    bulk_shirt_color,
                    bulk_shirt_type,
                    bulk_accessory,
                    bulk_broadcast_type,
                    bulk_emotion,
                    bulk_camera,
                    bulk_input_image_path
                ],
                outputs=bulk_output_message,
            )
        with gr.Tab(label="Browse Avatar Library"):
            refresh_gallery_button = gr.Button("Refresh Gallery")
            video_gallery = gr.Gallery(label="Video Gallery", value=get_video_gallery(), show_label=True, columns=5, allow_preview=True)
            gr.HTML("""
            <script>
                // Wait until the DOM is fully loaded
                document.addEventListener("DOMContentLoaded", function() {
                    // Add click event listener to all video elements in the gallery
                    document.querySelectorAll("video").forEach(function(video) {
                        video.addEventListener("click", function() {
                            // Pause all videos first
                            document.querySelectorAll("video").forEach(function(v) {
                                v.pause();
                            });
                            // Play the clicked video
                            video.play();
                        });
                    });
                });
            </script>
            """)
            refresh_gallery_button.click(fn=get_video_gallery, outputs=video_gallery)


if __name__ == "__main__":
    app.launch(inbrowser=True, show_api=False)
