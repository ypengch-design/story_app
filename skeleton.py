# function part
# img2text

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import io
import random

# Page Config
st.set_page_config(page_title="Kids Story Generator", # Browser tab title
                   page_icon="🧸")
st.title("🧸 AI Storyteller for Kids (Ages 3-10)")
st.write("Upload a picture, and I'll tell you a magical story!")

# Load Models
@st.cache_resource
def load_image_captioner():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_story_generator():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Helper Functions
def img2text(image):
    processor, model = load_image_captioner()
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def is_story_good(story_text, caption):
    """Check if the story is positive, contains key words from caption, and is long enough."""
    story_lower = story_text.lower()
    # Ban negative or scary words
    banned_words = ["scared", "afraid", "hate", "kill", "dead", "problem", "pain", "cry", "monster"]
    if any(w in story_lower for w in banned_words):
        return False
    # Must contain at least one significant noun from the caption
    caption_nouns = [w for w in caption.lower().split() if len(w) > 2]
    if not any(noun in story_lower for noun in caption_nouns):
        return False
    return True

def generate_story_internal(caption, tokenizer, model):
    """Attempt to generate a good story, up to 3 retries."""
    # Instruction that enforces happy, positive content
    prompt = (
        f"Write a very short, sweet, and happy story for little children aged 3-5. "
        f"The story must be exactly about: {caption}. "
        f"Only use friendly and positive words. Do NOT include anything scary or sad. "
        f"Describe what the characters do and how happy they feel. "
        f"The story should be 70 to 90 words long. End with a happy ending."
    )
    best_story = ""
    for attempt in range(3):
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.8,
        )
        raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # Remove possible prefixes the model may add
        for prefix in ["story:", "answer:"]:
            if raw.lower().startswith(prefix):
                raw = raw[len(prefix):].strip()
                break
        if is_story_good(raw, caption) and len(raw.split()) >= 30:
            best_story = raw
            break
        best_story = raw  
    return best_story

def text2story(caption):
    tokenizer, model = load_story_generator()
    raw_story = generate_story_internal(caption, tokenizer, model)

    # Keep only up to the last complete sentence
    for punct in ['.', '!', '?']:
        idx = raw_story.rfind(punct)
        if idx > 10:
            raw_story = raw_story[:idx+1]
            break
            
    # Capitalise the first letter
    if raw_story:
        raw_story = raw_story[0].upper() + raw_story[1:]

    # Length control
    words = raw_story.split()
    if len(words) < 50:
        happy_ends = [
            " And they all lived happily ever after.",
            " It was the best day ever, full of laughter and love!",
            " Everyone smiled and hugged, feeling safe and warm.",
        ]
        raw_story += random.choice(happy_ends)
    elif len(words) > 100:
        # If too long, truncate at the last full sentence within the first 100 words
        truncated = " ".join(words[:100])
        last_p = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_p > 10:
            raw_story = truncated[:last_p+1]
        else:
            raw_story = truncated + "..."
            
    # Make sure the story ends with punctuation
    if raw_story and not raw_story.endswith(('.', '!', '?')):
        raw_story += "."

    return raw_story

def text2audio(story_text):
    tts = gTTS(text=story_text, lang='en', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# Main part
uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Picture", use_container_width=True)

    # Button triggers the whole pipeline
    if st.button("✨ Generate Story"):
        with st.spinner("Creating a story just for you..."):
            # Step 1: Image captioning
            caption = img2text(image)
            st.subheader("📝 What I see in the picture")
            st.info(caption)

            # Step 2: Story generation
            story_text = text2story(caption)
            word_count = len(story_text.split())
            st.subheader(f"📖 Your Story ({word_count} words)")
            st.success(story_text)

            # Step 3: Text‑to‑speech
            audio_bytes = text2audio(story_text)
            st.subheader("🎧 Listen to the story")
            st.audio(audio_bytes, format="audio/mp3")
            
            st.balloons()
