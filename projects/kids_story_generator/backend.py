"""
backend.py

Groq + utilities for:
- Kids' Story Generator (Streamlit frontend)
- Story text (Groq), multi-language (English, Hindi, ...)
- Story image (Gemini Imagen API + Hugging Face fallback)
- Story voice (gTTS Google Text-to-Speech, MP3 files)
"""

import os
import tempfile
import base64
import requests
from groq import Groq
from dotenv import load_dotenv
from gtts import gTTS  # Google Text-to-Speech (unofficial)
from huggingface_hub import InferenceClient  # NEW

# ========= STEP 1: Environment & Clients =========

load_dotenv()  

# ----- Groq key -----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip().replace("GROQ_API_KEY=", "")

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    raise ValueError(
        "Invalid GROQ_API_KEY. "
        "Ensure .env has a line like:\n"
        "GROQ_API_KEY=gsk_your_real_key_here"
    )

groq_client = Groq(api_key=GROQ_API_KEY)

# ----- Gemini Image API -----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

IMAGEN_MODEL = "imagen-4.0-generate-001"
IMAGEN_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{IMAGEN_MODEL}:generateContent"
)

# ----- Hugging Face Image API via InferenceClient -----
HF_TOKEN = os.getenv("HF_TOKEN")  
hf_client = None
if HF_TOKEN:
    # Use router with nebius provider for FLUX.1-dev
    hf_client = InferenceClient(
        provider="nebius",
        api_key=HF_TOKEN,
    )  # [web:135]

HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"  # text-to-image model [web:135]

# Folder to store generated images
IMAGE_DIR = os.path.join(os.getcwd(), "generated_images")
os.makedirs(IMAGE_DIR, exist_ok=True)

# ========= STEP 2: Simple Age Config =========

AGE_CONFIG = {
    "3-5": {"min_words": 150, "max_words": 300, "vocab_level": "very_simple"},
    "6-8": {"min_words": 250, "max_words": 500, "vocab_level": "simple"},
    "9-12": {"min_words": 400, "max_words": 800, "vocab_level": "medium"},
}

# ========= STEP 2b: Language Config =========

LANGUAGE_CONFIG = {
    "English": {
        "code": "en",
        "label": "English",
        "instruction": "Write the entire story in natural, simple English.",
    },
    "Hindi": {
        "code": "hi",
        "label": "हिन्दी (Hindi)",
        "instruction": "पूरी कहानी स्वाभाविक और सरल हिन्दी में लिखें।",
    },
}

def get_age_config(age_group: str) -> dict:
    return AGE_CONFIG.get(age_group, AGE_CONFIG["6-8"])

def get_language_config(language_name: str) -> dict:
    return LANGUAGE_CONFIG.get(language_name, LANGUAGE_CONFIG["English"])

# ========= STEP 2c: Story Length Mapping =========

def get_length_range_from_slider(story_length: int, age_group: str) -> tuple[int, int]:
    base_cfg = get_age_config(age_group)
    center = story_length
    min_words = max(80, int(center * 0.7))
    max_words = int(center * 1.3)

    min_words = max(min_words, base_cfg["min_words"])
    max_words = min(max_words, base_cfg["max_words"])

    if min_words > max_words:
        min_words, max_words = base_cfg["min_words"], base_cfg["max_words"]

    return min_words, max_words

# ========= STEP 3: Build Story Prompt =========

def build_groq_prompt(
    age_group: str,
    keywords: str,
    child_name: str = "",
    language_name: str = "English",
    story_length: int = None,
    mood: str = None,
    hero_type: str = None,
) -> str:
    cfg = get_age_config(age_group)
    lang_cfg = get_language_config(language_name)

    if story_length is not None:
        min_words, max_words = get_length_range_from_slider(story_length, age_group)
    else:
        min_words, max_words = cfg["min_words"], cfg["max_words"]

    hero_line = ""
    if hero_type:
        if " " in hero_type:
            hero_core = hero_type.split(" ", 1)[1].strip()
        else:
            hero_core = hero_type
        hero_line = f"The main character should feel like a {hero_core}."

    name_line = (
        f"The main child character is named {child_name}." if child_name else ""
    )

    mood_line = ""
    if mood:
        mood_text = mood.split(" ", 1)[1].strip() if " " in mood else mood
        mood_line = (
            f"The overall mood of the story should feel {mood_text.lower()}, "
            f"but still safe and comforting for children."
        )

    return f"""
You are an award-winning children's story writer.

Write ONE imaginative, emotionally engaging story for a child aged {age_group}.

MANDATORY STORY STRUCTURE:
1. Opening: Introduce the main character and their everyday world.
2. Problem: Something unexpected goes wrong or a challenge appears.
3. Adventure: The character actively tries to solve the problem.
4. Climax: A tense or exciting moment where failure seems possible.
5. Resolution: The problem is solved in a satisfying and positive way.
6. Moral: End with ONE clear sentence that teaches a life lesson.

REQUIREMENTS:
- Story language: {language_name}. {lang_cfg["instruction"]}
- Vocabulary level: {cfg["vocab_level"]}, warm and child-friendly.
- Length: {min_words}–{max_words} words.
- Include these elements naturally: {keywords}
- {name_line}
- {hero_line}
- Include:
  - At least ONE short dialogue line in quotes.
  - Emotional descriptions (happy, scared, proud, excited).
  - A surprising but age-appropriate twist.
- {mood_line}

IMPORTANT:
- The story MUST have a strong ending where the child learns or changes.
- The moral must connect clearly to the story events.
- Do NOT use headings, lists, or emojis.

Return ONLY the story text.
""".strip()

# ========= STEP 4: Groq Text Generation =========

def generate_kids_story_groq(
    age_group: str,
    keywords: str,
    child_name: str = "",
    language_name: str = "English", 
    story_length: int = None,
    mood: str = None,
    hero_type: str = None,
) -> str:
    prompt = build_groq_prompt(
        age_group=age_group,
        keywords=keywords,
        child_name=child_name,
        language_name=language_name,
        story_length=story_length,
        mood=mood,
        hero_type=hero_type,
    )

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )

    story = completion.choices[0].message.content.strip()
    return story

# ========= STEP 5: Image Generation (Gemini + HF) =========

def build_image_prompt(
    age_group: str,
    keywords: str,
    mood: str = None,
    hero_type: str = None,
    child_name: str = "",
) -> str:
    mood_text = ""
    if mood:
        mood_core = mood.split(" ", 1)[1].strip() if " " in mood else mood
        mood_text = f"The mood is {mood_core.lower()}, "

    hero_text = ""
    if hero_type:
        if " " in hero_type:
            hero_core = hero_type.split(" ", 1)[1].strip()
        else:
            hero_core = hero_type
        hero_text = f"Show a child as a {hero_core}. "

    name_text = f"The child may be named {child_name}. " if child_name else ""

    return (
        f"Colorful 2D cartoon illustration for a kids' story for ages {age_group}. "
        f"Include these elements: {keywords}. "
        f"{hero_text}{name_text}{mood_text}"
        f"Use cute, friendly characters, bright colors, and a storybook style."
    )

def generate_story_image(prompt: str, image_path: str) -> str:
    if not GEMINI_API_KEY:
        print("[Gemini] No GEMINI_API_KEY set, skipping Gemini.")
        return ""

    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}

    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "image/png"},
    }

    try:
        resp = requests.post(
            IMAGEN_URL,
            params=params,
            json=data,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        print("[Gemini] request failed:", repr(e))
        return ""

    result = resp.json()
    candidates = result.get("candidates") or []
    if not candidates:
        print("[Gemini] no candidates returned")
        return ""

    parts = candidates[0].get("content", {}).get("parts") or []
    img_b64 = None
    for part in parts:
        inline_data = part.get("inline_data")
        if inline_data and inline_data.get("mime_type", "").startswith("image/"):
            img_b64 = inline_data.get("data")
            break

    if not img_b64:
        print("[Gemini] no inline image data")
        return ""

    img_bytes = base64.b64decode(img_b64)

    with open(image_path, "wb") as f:
        f.write(img_bytes)

    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        print("[Gemini] file not written correctly")
        return ""

    print("[Gemini] image saved at:", os.path.abspath(image_path))
    return image_path

# ---- Hugging Face image generation helper (InferenceClient) ----

def generate_story_image_hf(prompt: str, image_path: str) -> str:
    """
    Generate an image from text using Hugging Face InferenceClient
    (black-forest-labs/FLUX.1-dev). Saves image and returns image_path.
    """
    if not hf_client:
        print("[HF] No HF client (HF_TOKEN not set), skipping HF.")
        return ""

    try:
        # Returns a PIL.Image.Image
        img = hf_client.text_to_image(
            prompt,
            model=HF_MODEL_ID,
        )  # [web:135]
    except Exception as e:
        print("[HF] text_to_image failed:", repr(e))
        return ""

    try:
        img.save(image_path)
    except Exception as e:
        print("[HF] saving image failed:", repr(e))
        return ""

    if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
        print("[HF] file not written correctly")
        return ""

    print("[HF] image saved at:", os.path.abspath(image_path))
    print("[HF] size:", os.path.getsize(image_path), "bytes")
    return image_path

# ---- build image prompt directly from story text ----

def build_image_prompt_from_story(story: str) -> str:
    short_story = story[:400].replace("\n", " ")
    return (
        "Illustration for a children's story. Show the main child hero, key "
        "supporting characters, and the main setting from this scene in a colorful "
        "2D cartoon style, with bright colors and friendly faces. Focus on the "
        "central moment, not on text. Story excerpt: "
        f"{short_story}"
    )

# ========= STEP 6: Text-to-Speech (gTTS) =========

def save_story_audio_gtts(
    story_text: str,
    language_name: str = "English",
) -> str:
    lang_cfg = get_language_config(language_name)
    gtts_lang = lang_cfg["code"]

    tmp_dir = tempfile.gettempdir()
    audio_path = os.path.join(tmp_dir, "kids_story_audio.mp3")

    tts = gTTS(text=story_text, lang=gtts_lang, slow=False)
    tts.save(audio_path)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        return ""

    return audio_path

# ========= STEP 7: Combined helper =========

def generate_full_story_package(
    age_group: str,
    keywords: str,
    child_name: str = "",
    language_name: str = "English",
    story_length: int = None,
    mood: str = None,
    hero_type: str = None,
):
    story = generate_kids_story_groq(
        age_group=age_group,
        keywords=keywords,
        child_name=child_name,
        language_name=language_name,
        story_length=story_length,
        mood=mood,
        hero_type=hero_type,
    )

    img_prompt = build_image_prompt_from_story(story)

    image_path = os.path.join(IMAGE_DIR, "story_image.png")

    # Gemini first, HF fallback
    try:
        gen_path = generate_story_image(img_prompt, image_path=image_path)
        if gen_path:
            image_path = gen_path
        else:
            hf_path = generate_story_image_hf(img_prompt, image_path=image_path)
            image_path = hf_path or ""
    except Exception as e:
        print("[Image] both providers failed:", repr(e))
        try:
            hf_path = generate_story_image_hf(img_prompt, image_path=image_path)
            image_path = hf_path or ""
        except Exception as e2:
            print("[HF] fallback also failed:", repr(e2))
            image_path = ""

    try:
        audio_path = save_story_audio_gtts(story, language_name=language_name)
    except Exception as e:
        print("[Audio] gTTS failed:", repr(e))
        audio_path = ""

    return story, image_path, audio_path

# ========= STEP 8: Local smoke test =========

if __name__ == "__main__":
    print("=== Smoke test: Groq + Gemini/HF(FLUX) image + gTTS audio ===")
    try:
        demo_story, demo_img, demo_audio = generate_full_story_package(
            age_group="6-8",
            keywords="school, best friend, lost toy",
            child_name="Aarav",
            language_name="Hindi",
            story_length=300,
            mood="🎉 Exciting",
            hero_type="🦸 Superhero",
        )
        print("\n[OK] Story generated. First 300 chars:\n")
        print(demo_story[:300], "...\n")

        if demo_img:
            print(f"[OK] Image file created at: {demo_img}")
            print(f"Image file size: {os.path.getsize(demo_img)} bytes")
        else:
            print("[INFO] Image not generated (GEMINI_API_KEY/HF_TOKEN not set or API error).")

        if demo_audio:
            print(f"[OK] Audio file created at: {demo_audio}")
            print(f"File size: {os.path.getsize(demo_audio)} bytes")
        else:
            print("[INFO] Audio not generated (gTTS error or no internet).")

        print("\nSmoke test finished.")
    except Exception as e:
        print("\n[ERROR] Smoke test failed:")
        print(repr(e))
