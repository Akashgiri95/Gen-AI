"""
app.py


Enhanced Streamlit frontend for Kids’ Story Generator
With animations, interactive elements, and cartoon-style design
"""


import os
import streamlit as st
from backend import generate_full_story_package, AGE_CONFIG, LANGUAGE_CONFIG


# ========= STEP 1: Page Setup & Global Styles =========


st.set_page_config(
    page_title="✨ Kid's Story Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Comic+Neue:wght@400;700&display=swap');
:root {
    --primary: #FF6B8B;
    --secondary: #6ECBF5;
    --accent: #FFD166;
    --success: #06D6A0;
    --background: #FFF9F3;
}
* {
    font-family: 'Comic Neue', cursive;
}
.main {
    background: linear-gradient(135deg, #FFE6F2 0%, #E6F7FF 50%, #FFF9E6 100%);
    min-height: 100vh;
}
.stApp {
    background: linear-gradient(135deg, #FFE6F2 0%, #E6F7FF 50%, #FFF9E6 100%);
}
.magic-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Fredoka One', cursive;
    font-size: 3.5rem;
    margin-bottom: 10px;
    text-shadow: 3px 3px 0px rgba(255,255,255,0.5);
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
.story-card {
    background: white;
    border-radius: 25px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(255,107,139,0.1);
    border: 4px solid var(--accent);
    margin: 20px 0;
    transition: transform 0.3s ease;
}
.story-card:hover {
    transform: translateY(-5px);
}
.stButton > button {
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    border-radius: 50px;
    padding: 15px 40px;
    font-size: 1.2rem;
    font-weight: bold;
    font-family: 'Fredoka One', cursive;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(255,107,139,0.3);
    width: 100%;
}
.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 8px 20px rgba(255,107,139,0.4);
}
.stButton > button:active {
    transform: scale(0.98);
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border-radius: 20px;
    border: 3px solid var(--accent);
    padding: 15px;
    font-size: 1.1rem;
}
.stSelectbox > div > div > div {
    border-radius: 20px;
    border: 3px solid var(--accent);
}
.age-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--secondary), #8AE1FC);
    color: white;
    font-weight: bold;
    font-size: 1rem;
    margin: 5px;
    animation: pulse 2s infinite;
}
.lang-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--primary), #FF8BA7);
    color: white;
    font-weight: bold;
    font-size: 1rem;
    margin: 5px;
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
.story-text {
    background: linear-gradient(135deg, #FFF9F4, #FFFAF0);
    border-radius: 20px;
    padding: 25px;
    border: 3px dashed var(--accent);
    font-size: 1.2rem;
    line-height: 1.8;
    box-shadow: inset 0 0 20px rgba(255,214,102,0.1);
}
.floating-emoji {
    font-size: 2rem;
    animation: float 6s ease-in-out infinite;
    display: inline-block;
    margin: 0 10px;
}
.stProgress > div > div > div > div {
    background-color: var(--success);
}
.stRadio > div {
    background-color: white;
    border-radius: 20px;
    padding: 10px;
    border: 3px solid var(--accent);
}
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ========= STEP 2: Header with Animated Emojis =========


col1, col2, col3 = st.columns([1, 2, 1])


with col2:
    st.markdown('<h1 class="magic-header">✨ Kids Story Generator 🧚‍♂️</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <span class="floating-emoji" style="animation-delay: 0s;">📚</span>
        <span class="floating-emoji" style="animation-delay: 1s;">🌈</span>
        <span class="floating-emoji" style="animation-delay: 2s;">🐉</span>
        <span class="floating-emoji" style="animation-delay: 3s;">🧸</span>
        <span class="floating-emoji" style="animation-delay: 4s;">🚀</span>
        <span class="floating-emoji" style="animation-delay: 5s;">🎨</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 30px;'>
    Create magical bedtime stories with pictures and voices! Perfect for young dreamers and adventurers. 🌟
    </p>
    """, unsafe_allow_html=True)


# ========= STEP 3: Sidebar (hero, length, mood) =========


with st.sidebar:
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 20px; border: 4px solid #FFD166;">
        <h3 style="color: #FF6B8B; text-align: center;">🎭 Choose Your Hero</h3>
    </div>
    """, unsafe_allow_html=True)


    character_type = st.radio(
        "What kind of hero is your child?",
        ["👸 Princess/Prince", "🦸 Superhero", "🧙 Wizard/Witch", "🐱 Animal Friend", "👽 Space Explorer"],
        index=0,
    )


    st.markdown("---")


    story_length = st.slider(
        "Story Length (controls word count)",
        min_value=100,
        max_value=500,
        value=200,
        step=50,
    )


    st.markdown("---")


    mood = st.select_slider(
        "Story Mood",
        options=["😴 Calm", "😊 Happy", "🎉 Exciting", "🤔 Mysterious", "🏆 Adventurous"],
        value="😊 Happy",
    )


# ========= STEP 4: Main Content =========


with st.container():
    st.markdown('<div class="story-card">', unsafe_allow_html=True)


    col_left, col_right = st.columns([1, 1])


    age_groups = list(AGE_CONFIG.keys())
    language_names = list(LANGUAGE_CONFIG.keys())


    with col_left:
        selected_age_group = st.selectbox(
            "🎂 Select Age Group",
            age_groups,
            index=0,
            help="Choose the age group for story complexity",
        )
        st.markdown(
            f'<div style="text-align: center;"><span class="age-badge">For {selected_age_group}</span></div>',
            unsafe_allow_html=True,
        )


        child_name = st.text_input(
            "🦸 Hero's Name",
            value="",
            placeholder="Enter your hero's name...",
            help="The main character of the story",
        )
        if child_name:
            st.success(f"✨ {child_name} is ready for adventure!")


    with col_right:
        selected_language = st.selectbox(
            "🌍 Story Language",
            language_names,
            index=0,
            help="Choose the language for your story",
        )
        lang_label = LANGUAGE_CONFIG[selected_language]["label"]
        st.markdown(
            f'<div style="text-align: center;"><span class="lang-badge">{lang_label}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align: center; margin-top: 20px;"><span class="lang-badge" style="background: linear-gradient(90deg, #06D6A0, #0AD7B7);">{mood}</span></div>',
            unsafe_allow_html=True,
        )


# ========= STEP 5: Keywords Input =========


st.markdown("""
<div style="margin: 30px 0;">
    <h3 style="color: #FF6B8B; text-align: center;">✨ Magical Ingredients for Your Story</h3>
    <p style="text-align: center; color: #666;">Mix these words to create something special!</p>
</div>
""", unsafe_allow_html=True)


keywords = st.text_area(
    "Keywords",
    value="",
    placeholder="Type magic words like: dragon, rainbow, castle, spaceship, treasure, friend, forest...",
    height=120,
    help="Separate keywords with commas for best results",
)


# ========= STEP 6: Generate Button =========


center1, center2, center3 = st.columns([1, 2, 1])
with center2:
    generate_btn = st.button("🎭 Generate Magical Story!", use_container_width=True, type="primary")


# ========= STEP 7: Story Generation & Display =========


if generate_btn:
    if not keywords.strip():
        st.error("🎨 Please add some magic words to create your story!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()


        status_text.text("🧚 Preparing magic ingredients...")
        progress_bar.progress(25)


        status_text.text("📖 Weaving your magical tale...")
        progress_bar.progress(50)


        try:
            story, image_path, audio_path = generate_full_story_package(
                age_group=selected_age_group,
                keywords=keywords.strip(),
                child_name=child_name.strip(),
                language_name=selected_language,
                story_length=story_length,
                mood=mood,
                hero_type=character_type,
            )


            status_text.text("🎨 Painting magical pictures...")
            progress_bar.progress(75)


            status_text.text("🎤 Adding gTTS voice...")
            progress_bar.progress(100)


            status_text.text("✨ Your magical story is ready!")
            st.balloons()
            st.success("🎉 Your magical story has been created!")


            st.markdown("""
            <div style="margin: 40px 0 20px 0;">
                <h2 style="color: #FF6B8B; text-align: center; border-bottom: 3px solid #FFD166; padding-bottom: 10px;">
                    📖 Your Magical Story
                </h2>
            </div>
            """, unsafe_allow_html=True)


            if child_name:
                st.markdown(
                    f"""
                    <div style="text-align: center; margin-bottom: 20px; padding: 15px; background: linear-gradient(135deg, #FFF0F7, #F0F8FF); border-radius: 20px;">
                        <h3 style="color: #6ECBF5;">Meet {child_name}, the {character_type.split()[1]}!</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


            st.markdown(
                f'<div class="story-text">{story.replace("\\n", "<br><br>")}</div>',
                unsafe_allow_html=True,
            )


            # Illustration
            if image_path and os.path.exists(image_path):
                st.markdown("""
                <div style="margin: 40px 0 20px 0;">
                    <h2 style="color: #FF6B8B; text-align: center; border-bottom: 3px solid #FFD166; padding-bottom: 10px;">
                        🎨 Story Illustration
                    </h2>
                </div>
                """, unsafe_allow_html=True)


                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.image(
                        image_path,
                        use_column_width=True,
                        caption=f"A magical scene from {child_name if child_name else 'the story'}'s adventure",
                    )
            else:
                st.info("No illustration was generated. Check your GEMINI_API_KEY / HF_TOKEN or API limits.")


            # Audio
            if audio_path and os.path.exists(audio_path):
                st.markdown("""
                <div style="margin: 40px 0 20px 0;">
                    <h2 style="color: #FF6B8B; text-align: center; border-bottom: 3px solid #FFD166; padding-bottom: 10px;">
                        🎧 Listen to the Story
                    </h2>
                </div>
                """, unsafe_allow_html=True)


                try:
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()
                    c1, c2, c3 = st.columns([1, 3, 1])
                    with c2:
                        st.audio(audio_bytes, format="audio/mpeg")
                        st.markdown("""
                        <div style="text-align: center; margin-top: 10px;">
                            <span style="color: #666; font-size: 0.9rem;">
                                🔊 Click play to hear your story narrated
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Could not load audio: {e}")
            else:
                st.info("Narration not available. Check your internet connection for gTTS.")


            # Download and share
            st.markdown("""
            <div style="margin: 40px 0 20px 0;">
                <h2 style="color: #FF6B8B; text-align: center; border-bottom: 3px solid #FFD166; padding-bottom: 10px;">
                    📥 Save Your Magical Story
                </h2>
            </div>
            """, unsafe_allow_html=True)


            c1, c2, c3 = st.columns(3)


            with c1:
                file_content = f"""✨ MAGICAL STORY ✨


Hero: {child_name if child_name else "Brave Adventurer"}
Age Group: {selected_age_group}
Language: {selected_language}
Mood: {mood}
Character: {character_type}


{story}


Created with Magic Story Generator
"""
                st.download_button(
                    label="📄 Download as Text",
                    data=file_content,
                    file_name=f"magical_story_{child_name if child_name else 'adventure'}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )


            with c2:
                st.button(
                    "🔗 Share Story",
                    use_container_width=True,
                    help="Copy text below and share via WhatsApp, email, etc.",
                )


            with c3:
                st.button(
                    "🖨️ Print Story",
                    use_container_width=True,
                    help="Use your browser's print option to print this page.",
                )


            st.markdown("""
            <div style="margin-top: 30px;">
                <h4 style="color: #FF6B8B;">Copy and Share</h4>
                <p style="color: #666;">Copy the text below to share on WhatsApp, email, or social media:</p>
            </div>
            """, unsafe_allow_html=True)


            st.text_area(
                "Story Text",
                value=story,
                height=200,
            )


        except Exception as e:
            st.error(f"Oops! Something went wrong: {str(e)}")
            st.info("Please check your API keys and internet connection, then try again.")


    st.markdown("</div>", unsafe_allow_html=True)


# ========= STEP 8: Footer =========


st.markdown("---")


fc1, fc2, fc3 = st.columns(3)


with fc1:
    st.markdown("""
    <div style="text-align: center;">
        <h4>🌟 Tips</h4>
        <p style="font-size: 0.9rem; color: #666;">
        • Use descriptive words<br>
        • Mix different themes<br>
        • Add your child's name<br>
        • Try different languages
        </p>
    </div>
    """, unsafe_allow_html=True)


with fc2:
    st.markdown("""
    <div style="text-align: center;">
        <h4>📚 Story Ideas</h4>
        <p style="font-size: 0.9rem; color: #666%;">
        • Space adventure with robot<br>
        • Forest mystery with animals<br>
        • Underwater treasure hunt<br>
        • Castle in the clouds
        </p>
    </div>
    """, unsafe_allow_html=True)


with fc3:
    st.markdown("""
    <div style="text-align: center;">
        <h4>🎨 Features</h4>
        <p style="font-size: 0.9rem; color: #666%;">
        • AI-generated stories<br>
        • Beautiful illustrations<br>
        • Voice narration (gTTS)<br>
        • Multiple languages<br>
        • Age-appropriate content
        </p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.5); border-radius: 20px;">
    <p style="color: #888; font-size: 0.9rem;">
    Made with ❤️ for young dreamers everywhere | Perfect for bedtime stories, learning, and imagination adventures
    </p>
</div>
""", unsafe_allow_html=True)