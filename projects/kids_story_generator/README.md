# Kid's Story Generator

A Streamlit app that generates complete, age-appropriate children's stories —
text, an illustration, and narrated audio — from a single prompt, with
configurable age group, language, and story length.

## How it works

- **`AGE_CONFIG`** — defines vocabulary complexity, tone, and target length ranges
  per age band (e.g. 3-5, 6-8, 9-12), so the same prompt produces a simpler story
  for younger readers and a richer one for older readers.
- **`LANGUAGE_CONFIG`** — supports multiple output languages (English, Hindi, ...).
- **`build_groq_prompt` / `generate_kids_story_groq`** — constructs a prompt
  incorporating age, language, and length constraints, then generates the story
  text via the **Groq** API.
- **`build_image_prompt` / `build_image_prompt_from_story`** — derives an
  illustration prompt from the generated story.
- **`generate_story_image` / `generate_story_image_hf`** — generates a story
  illustration via **Gemini Imagen**, with a **Hugging Face** inference fallback if
  the primary call fails.
- **`save_story_audio_gtts`** — narrates the story using **gTTS** (Google
  Text-to-Speech), producing an audio file (`story_audio.wav` is a sample output).
- **`generate_full_story_package`** — orchestrates all of the above into a single
  call: prompt in → {story text, illustration, audio} out.

## UI (`app.py`)

A cartoon-styled Streamlit frontend (custom fonts/CSS) where a parent/child picks
an age group, language, and story length, then receives the generated story,
image, and playable audio narration.

## Run it

```bash
pip install -r requirements.txt   # streamlit, groq, gtts, huggingface_hub, python-dotenv
# add GROQ_API_KEY / HF token / Gemini key to a .env file
streamlit run app.py
```

## Key Takeaways

- Combining three different generative modalities (text, image, audio) behind one
  orchestration function (`generate_full_story_package`) keeps the Streamlit UI
  layer simple — it just calls one function and renders the result.
- Building in a fallback image provider (Gemini → Hugging Face) is a practical
  pattern for apps depending on external APIs with rate limits or occasional
  failures.
- Parameterizing prompts by age group and language, rather than hardcoding a
  single style, is what turns a "story generator" into a genuinely reusable tool.

## Tech Stack

Streamlit · Groq API · Gemini Imagen · Hugging Face Inference · gTTS
