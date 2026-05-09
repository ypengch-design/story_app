# function part
# img2text

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from huggingface_hub import InferenceClient
from gtts import gTTS
import io
import requests
import random

# --------------------- Page Configuration ---------------------
st.set_page_config(page_title="Magic Storyteller", page_icon="📖")
st.title("🌟 AI Magic Storyteller")
st.write("Upload a photo and I will tell you a wonderful fairy tale!")

# --------------------- Load Image Captioning Model ---------------------
@st.cache_resource
def load_image_captioner():
    """Load BLIP processor and model for image-to-text captioning."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# --------------------- Story Generation via API ---------------------
def generate_story_via_api(caption, token):
    """Call Hugging Face InferenceClient to generate a children's story."""
    client = InferenceClient(token=token)
    prompt = (
        f"[INST] Write a very short, sweet story for children aged 3-5. "
        f"The story must be exactly about: {caption}. "
        f"Use simple words. It must be 60 to 90 words. "
        f"Describe what happens and how the characters feel. "
        f"End with a happy sentence. Write in third person. [/INST]"
    )
    try:
        # 使用最新的文本生成模型，可尝试多个
        response = client.text_generation(
            prompt,
            model="mistralai/Mistral-7B-Instruct-v0.1",   # 如果还不行就换下面注释的那个
            # model="HuggingFaceH4/zephyr-7b-beta",      # 备选
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
        )
        return response.strip()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return None

# --------------------- Post-process Story ---------------------
def post_process_story(raw_story, caption):
    """Clean up story text, ensure 50-100 words and a happy ending."""
    story = raw_story.replace('"', '').strip()
    if story.lower().startswith("once upon a time"):
        story = story[0].upper() + story[1:]

    # Keep only up to last full sentence
    for punct in ['.', '!', '?']:
        idx = story.rfind(punct)
        if idx > 10:
            story = story[:idx+1]
            break

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

# --------------------- Streamlit UI ---------------------
uploaded_file = st.file_uploader("📸 Choose a photo...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your Uploaded Image", use_container_width=True)

    # Check for Hugging Face token in secrets
    hf_token = st.secrets.get("HF_TOKEN", None)
    if hf_token is None:
        st.error("⚠️ Please set your Hugging Face token in Streamlit Cloud Secrets as `HF_TOKEN`.")
        st.stop()

    if st.button("✨ Generate Magic Story"):
        with st.spinner("Writing your fairy tale..."):
            # 1. Image captioning
            processor, blip_model = load_image_captioner()
            inputs = processor(image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            st.subheader("📝 Image Caption")
            st.info(caption)

            # 2. Story generation via API
            raw_story = generate_story_via_api(caption, hf_token)
            if raw_story is None:
                st.error("Story generation failed. Please try again.")
                st.stop()

            story_text = post_process_story(raw_story, caption)
            word_count = len(story_text.split())

            # 3. Text-to-speech
            audio_bytes = text2audio(story_text)

            # 4. Display results
            st.subheader(f"📖 Your Magic Story ({word_count} words)")
            st.success(story_text)
            st.subheader("🎙️ Listen to the Story")
            st.audio(audio_bytes, format="audio/mp3")
            st.balloons()
