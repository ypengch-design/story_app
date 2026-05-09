# function part
# img2text

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
import io
import random

# --------------------- Page Config ---------------------
st.set_page_config(page_title="Kids Story Generator", page_icon="🧸")
st.title("🧸 AI Storyteller for Kids (Ages 3-10)")
st.write("Upload a picture, and I'll tell you a magical story!")

# --------------------- Model Loading ---------------------
@st.cache_resource
def load_image_captioner():
    """Load BLIP model for image captioning (processor + model)."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_story_generator():
    """Load text2text-generation model (Flan-T5 base)."""
    return pipeline("text2text-generation", model="google/flan-t5-base")

# --------------------- Core Functions ---------------------
def img2text(image):
    """Convert a PIL Image to a text caption."""
    processor, model = load_image_captioner()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def text2story(caption):
    """Generate a 50-100 word children's story from the caption."""
    generator = load_story_generator()

    prompt = (
        f"Write a very short, sweet story for children aged 3-5. "
        f"The story must be exactly about: {caption}. "
        f"Use simple words. Make the story 60 to 90 words long. "
        f"Describe what happens, how the characters feel, and end with a happy sentence."
    )

    outputs = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.5,
    )

    raw_story = outputs[0]["generated_text"].strip()

    # Clean potential artifacts
    if raw_story.lower().startswith("story:"):
        raw_story = raw_story[6:].strip()
    elif raw_story.lower().startswith("answer:"):
        raw_story = raw_story[7:].strip()
    # In rare cases, the model echoes the prompt; remove it
    if raw_story.lower().startswith(prompt.lower()):
        raw_story = raw_story[len(prompt):].strip()

    # Keep only up to last complete sentence
    for punct in ['.', '!', '?']:
        idx = raw_story.rfind(punct)
        if idx > 10:
            raw_story = raw_story[:idx+1]
            break

    if raw_story:
        raw_story = raw_story[0].upper() + raw_story[1:]

    # Ensure word count 50-100
    words = raw_story.split()
    if len(words) < 50:
        happy_ends = [
            " And they all lived happily ever after.",
            " It was the best day ever, full of laughter and love!",
            " Everyone smiled and hugged, feeling safe and warm.",
        ]
        raw_story += random.choice(happy_ends)
    elif len(words) > 100:
        truncated = " ".join(words[:100])
        last_p = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_p > 10:
            raw_story = truncated[:last_p+1]
        else:
            raw_story = truncated + "..."

    if raw_story and not raw_story.endswith(('.', '!', '?')):
        raw_story += "."

    return raw_story

def text2audio(story_text):
    """Convert story text to MP3 audio (BytesIO)."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# --------------------- Streamlit UI ---------------------
uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Picture", use_container_width=True)

    if st.button("✨ Generate Story"):
        with st.spinner("Creating a story just for you..."):
            caption = img2text(image)
            st.subheader("📝 What I see in the picture")
            st.info(caption)

            story_text = text2story(caption)
            word_count = len(story_text.split())
            st.subheader(f"📖 Your Story ({word_count} words)")
            st.success(story_text)

            audio_bytes = text2audio(story_text)
            st.subheader("🎧 Listen to the story")
            st.audio(audio_bytes, format="audio/mp3")

            st.balloons()
