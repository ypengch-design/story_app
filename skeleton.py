# function part
# img2text

import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS
import io
import requests
import random

# --------------------- Page Configuration ---------------------
st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
st.title("🌟 AI Magic Storyteller")
st.write("Upload a photo and I will tell you a wonderful fairy tale!")

# --------------------- Local Models ---------------------
@st.cache_resource
def load_image_captioner():
    """Load the image captioning model (Salesforce BLIP)."""
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

captioner = load_image_captioner()   # We keep BLIP locally for image description

# --------------------- API Story Generation ---------------------
def generate_story_via_api(caption, token):
    """
    Call Hugging Face Inference API to generate a children's story.
    Using mistralai/Mistral-7B-Instruct-v0.1 for excellent instruction following.
    """
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Authorization": f"Bearer {token}"}

    prompt = (
        f"[INST] Write a very short, sweet story for children aged 3-5. "
        f"The story must be exactly about: {caption}. "
        f"Use simple words. It must be 60 to 90 words. "
        f"Describe what happens and how the characters feel. "
        f"End with a happy sentence. Write in third person. [/INST]"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.9,
            "do_sample": True,
            "return_full_text": False,
        }
    }

    # Try API call
    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            story = result[0]["generated_text"].strip()
            return story
        else:
            st.error("Unexpected API response format.")
            return None
    else:
        st.error(f"API error: {response.status_code} - {response.text}")
        return None

def post_process_story(raw_story, caption):
    """
    Clean up the story, ensure full sentences, add a random happy ending if needed.
    """
    # Remove possible quotation marks
    story = raw_story.replace('"', '').strip()

    # If it already starts with "Once upon a time", capitalize
    if story.lower().startswith("once upon a time"):
        story = story[0].upper() + story[1:]

    # Truncate to the last full sentence
    for p in ['.', '!', '?']:
        idx = story.rfind(p)
        if idx > 10:
            story = story[:idx+1]
            break

    # Ensure word count 50-100
    words = story.split()
    if len(words) < 50:
        happy_ends = [
            " And they all lived happily ever after.",
            " It was the best day ever, full of laughter and love!",
            " Everyone smiled and hugged, feeling safe and warm.",
        ]
        story += random.choice(happy_ends)
    elif len(words) > 100:
        truncated = " ".join(words[:100])
        last_p = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_p > 10:
            story = truncated[:last_p+1]
        else:
            story = truncated + "..."

    # Ensure ending punctuation
    if story and not story.endswith(('.', '!', '?')):
        story += "."

    return story

# --------------------- Text-to-Speech ---------------------
def text2audio(story_text):
    """Convert story text to MP3 audio."""
    tts = gTTS(text=story_text, lang='en', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# --------------------- Main App ---------------------
uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Uploaded Image", use_container_width=True)

    # Check if Hugging Face token is available in secrets
    hf_token = st.secrets.get("HF_TOKEN", None)

    if hf_token is None:
        st.error("⚠️ Please set your Hugging Face token in Streamlit Cloud Secrets as `HF_TOKEN`.")
        st.stop()

    if st.button("✨ Generate Magic Story"):
        with st.spinner("Writing your fairy tale..."):
            # 1. Get image caption
            caption = captioner(image)[0]["generated_text"]
            st.subheader("📝 Image Caption")
            st.info(caption)

            # 2. Generate story via API
            raw_story = generate_story_via_api(caption, hf_token)
            if raw_story is None:
                st.error("Story generation failed. Please try again.")
                st.stop()

            story_text = post_process_story(raw_story, caption)
            word_count = len(story_text.split())

            # 3. Convert to audio
            audio_bytes = text2audio(story_text)

            # 4. Display
            st.subheader(f"📖 Your Magic Story ({word_count} words)")
            st.success(story_text)

            st.subheader("🎙️ Listen to the Story")
            st.audio(audio_bytes, format="audio/mp3")

            st.balloons()