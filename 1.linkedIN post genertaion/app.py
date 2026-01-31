import os
import tempfile
import urllib.parse
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
from streamlit.components.v1 import html
from backend import LinkedInPostGenerator, extract_text_from_pdf

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="LinkedIn Post Generator Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0A66C2; /* LinkedIn Blue */
        --secondary: #004182;
        --accent: #00A0DC;
        --success: #057642;
        --warning: #E6A700;
        --danger: #C9372C;
        --light: #F3F6F8;
        --dark: #1D2228;
    }
    
    /* Global styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    
    /* Professional header */
    .professional-header {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        padding: 25px 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(10, 102, 194, 0.15);
    }
    
    .professional-header h1 {
        color: white;
        font-size: 2.8rem;
        margin: 0;
        font-weight: 700;
    }
    
    .professional-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 10px 0 0 0;
    }
    
    /* Cards */
    .professional-card {
        background: white;
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.08);
        border-left: 4px solid var(--primary);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .professional-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(10, 102, 194, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(10, 102, 194, 0.3);
    }
    
    .primary-button {
        background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e1e8ed;
        padding: 12px;
        font-size: 14px;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.1);
    }
    
    /* Metrics */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border-top: 3px solid var(--primary);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin: 10px 0;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    
    .status-ready {
        background: #D1F7C4;
        color: var(--success);
    }
    
    .status-scheduled {
        background: #C2E7FF;
        color: var(--primary);
    }
    
    .status-pending {
        background: #FFF2CC;
        color: var(--warning);
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary);
    }
</style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
if "generator" not in st.session_state:
    st.session_state.generator = LinkedInPostGenerator()
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "posts" not in st.session_state:
    st.session_state.posts = []
if "schedule" not in st.session_state:
    st.session_state.schedule = []
if "num_posts" not in st.session_state:
    st.session_state.num_posts = 3
if "max_words" not in st.session_state:
    st.session_state.max_words = 200
if "post_times" not in st.session_state:
    st.session_state.post_times = ["09:00", "12:00", "15:00", "18:00"]
if "generated_contents" not in st.session_state:
    st.session_state.generated_contents = {}
if "revision_version" not in st.session_state:
    st.session_state.revision_version = {}
if "auto_schedule" not in st.session_state:
    st.session_state.auto_schedule = False

# ============ HELPER FUNCTIONS ============
def open_linkedin_with_text(text: str):
    encoded_text = urllib.parse.quote_plus(text)
    url = f"https://www.linkedin.com/feed/?shareActive=true&mini=true&text={encoded_text}"
    script = f"""
        <script type="text/javascript">
            window.open("{url}", "_blank");
        </script>
    """
    html(script)

def validate_word_count(text: str, max_words: int) -> tuple:
    """Validate and count words in text"""
    words = text.split()
    word_count = len(words)
    is_valid = word_count <= max_words + 20  # Allow small buffer
    return word_count, is_valid

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown('<div class="professional-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Generation Settings")
    
    # Number of posts
    st.markdown("**Number of Posts**")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("−", key="minus_posts"):
            st.session_state.num_posts = max(1, st.session_state.num_posts - 1)
    with col2:
        st.session_state.num_posts = st.number_input(
            label="Posts",
            min_value=1,
            max_value=10,
            value=st.session_state.num_posts,
            label_visibility="collapsed",
            key="num_posts_input"
        )
    with col3:
        if st.button("+", key="plus_posts"):
            st.session_state.num_posts = min(10, st.session_state.num_posts + 1)
    
    st.markdown("---")
    
    # Word limit with strict enforcement
    st.markdown("**Word Limit per Post**")
    st.session_state.max_words = st.slider(
        "Max words (strictly enforced):",
        min_value=50,
        max_value=500,
        value=st.session_state.max_words,
        step=10,
        help="Posts will be strictly kept within ±20 words of this limit"
    )
    
    st.markdown(f"""
    <div style="background: #E8F4FD; padding: 10px; border-radius: 8px; margin: 10px 0;">
        <small>📏 Target: <strong>{max(30, st.session_state.max_words - 20)} - {st.session_state.max_words + 20}</strong> words per post</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Posting times for automation
    st.markdown("**Auto-Posting Times**")
    time1 = st.time_input("Time 1", value=datetime.strptime("09:00", "%H:%M").time())
    time2 = st.time_input("Time 2", value=datetime.strptime("12:00", "%H:%M").time())
    time3 = st.time_input("Time 3", value=datetime.strptime("15:00", "%H:%M").time())
    time4 = st.time_input("Time 4", value=datetime.strptime("18:00", "%H:%M").time())
    
    st.session_state.post_times = [
        time1.strftime("%H:%M"),
        time2.strftime("%H:%M"),
        time3.strftime("%H:%M"),
        time4.strftime("%H:%M")
    ]
    
    st.markdown("---")
    
    # Auto-schedule toggle
    st.session_state.auto_schedule = st.checkbox(
        "Enable Auto-Scheduling",
        value=st.session_state.auto_schedule,
        help="Automatically schedule posts for LinkedIn posting"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============ MAIN HEADER ============
st.markdown("""
<div class="professional-header">
    <h1>🚀 LinkedIn Post Generator Pro</h1>
    <p>AI-powered content creation with RAG + CrewAI | Auto-scheduling | Professional optimization</p>
</div>
""", unsafe_allow_html=True)

# ============ TABS ============
tab_main, tab_schedule, tab_stats, tab_automation = st.tabs(
    ["📄 Generate Posts", "📅 Schedule", "📊 Analytics", "🤖 Automation"]
)

# ============ TAB 1: GENERATE POSTS ============
with tab_main:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Resume (PDF)")
        pdf_file = st.file_uploader("Choose PDF file", type=["pdf"], key="pdf_uploader")
        if pdf_file is not None:
            if st.button("📥 Process PDF", type="primary", use_container_width=True):
                with st.spinner("Extracting text from PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.read())
                        tmp_path = tmp.name
                    text = extract_text_from_pdf(tmp_path)
                    os.unlink(tmp_path)
                    if text.strip():
                        st.session_state.resume_text = text
                        # Clear previous content
                        st.session_state.generated_contents = {}
                        st.session_state.revision_version = {}
                        st.success("✅ PDF processed successfully!")
                        
                        # Show statistics
                        word_count = len(text.split())
                        st.info(f"""
                        **Extracted Statistics:**
                        - Words: {word_count:,}
                        - Characters: {len(text):,}
                        - Estimated pages: {word_count // 250 + 1}
                        """)
                    else:
                        st.error("❌ Could not extract text from PDF. Please try another file.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Or Paste Text")
        txt = st.text_area(
            "Paste your resume or topic text here:",
            value=st.session_state.resume_text,
            height=200,
            key="text_input",
            placeholder="Paste your resume text here...\n\nExample:\nJohn Doe - Software Engineer\n5+ years experience in Python, AWS, Docker\nLed team of 5 developers...",
        )
        if st.button("💾 Use This Text", use_container_width=True):
            if txt.strip():
                st.session_state.resume_text = txt
                # Clear previous content
                st.session_state.generated_contents = {}
                st.session_state.revision_version = {}
                st.success("✅ Text saved for generation!")
                st.info(f"Text length: {len(txt.split()):,} words")
            else:
                st.warning("⚠️ Please enter some text first.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generation Section
    if not st.session_state.resume_text:
        st.info("👆 Please upload a PDF or paste text to get started.")
    else:
        col_gen1, col_gen2 = st.columns([3, 1])
        with col_gen1:
            if st.button("🚀 Generate All Posts", type="primary", use_container_width=True):
                with st.spinner("🤖 Running AI pipeline (RAG + CrewAI)..."):
                    try:
                        posts = st.session_state.generator.run_full_pipeline(
                            resume_text=st.session_state.resume_text,
                            num_posts=st.session_state.num_posts,
                            max_words=st.session_state.max_words,
                        )
                        st.session_state.posts = posts
                        
                        # Initialize generated contents
                        st.session_state.generated_contents = {}
                        for idx, p in enumerate(posts):
                            st.session_state.generated_contents[idx] = p["content"]
                            st.session_state.revision_version[idx] = 1
                        
                        st.success(f"✅ Successfully generated {len(posts)} posts!")
                        
                        # Auto-schedule if enabled
                        if st.session_state.auto_schedule and posts:
                            start_date = datetime.now().strftime("%Y-%m-%d")
                            scheduled = st.session_state.generator.auto_schedule_posts(
                                start_date=start_date,
                                post_times=st.session_state.post_times
                            )
                            if scheduled:
                                st.info(f"📅 Auto-scheduled {len(posts)} posts!")
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {str(e)}")
        
        with col_gen2:
            # Quick stats
            if st.session_state.posts:
                total_words = sum(p["word_count"] for p in st.session_state.posts)
                avg_words = total_words // len(st.session_state.posts)
                st.metric("Avg Words/Post", avg_words)
        
        # Display Generated Posts
        if st.session_state.posts:
            st.markdown("---")
            st.markdown("### 📋 Generated Posts")
            
            for idx, post in enumerate(st.session_state.posts):
                revision_ver = st.session_state.revision_version.get(idx, 1)
                
                st.markdown(f'<div class="professional-card">', unsafe_allow_html=True)
                
                # Post header with metrics
                col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
                with col_h1:
                    st.markdown(f"#### 📝 Day {post.get('day', idx+1)}: **{post.get('topic_title', '')}**")
                with col_h2:
                    word_count = post["word_count"]
                    color = "green" if post.get("word_limit_met", False) else "orange"
                    st.markdown(f'<span style="color: {color}; font-weight: bold;">📊 {word_count} words</span>', unsafe_allow_html=True)
                with col_h3:
                    quality = post.get("quality_score", 0)
                    st.markdown(f'<span style="color: #0A66C2; font-weight: bold;">⭐ {quality}/100</span>', unsafe_allow_html=True)
                
                # Post content
                current_content = st.session_state.generated_contents.get(idx, post["content"])
                textarea_key = f"post_content_{idx}_v{revision_ver}"
                
                content = st.text_area(
                    "Post Content:",
                    value=current_content,
                    height=150,
                    key=textarea_key,
                    help="Edit the post content directly here"
                )
                
                # Update session state
                st.session_state.posts[idx]["content"] = content
                st.session_state.posts[idx]["word_count"] = len(content.split())
                
                # Action buttons
                col_b1, col_b2, col_b3 = st.columns(3)
                
                with col_b1:
                    # Regenerate with feedback
                    with st.expander("🔄 Regenerate with Feedback"):
                        fb = st.text_area(
                            "Your feedback:",
                            placeholder="E.g., 'Make it more technical' or 'Focus on leadership aspects'",
                            key=f"fb_{idx}",
                            height=80
                        )
                        if st.button("🔄 Regenerate", key=f"regenerate_{idx}"):
                            if fb.strip():
                                with st.spinner("Revising post..."):
                                    updated = st.session_state.generator.revise_post(
                                        index=idx,
                                        feedback=fb,
                                        max_words=st.session_state.max_words,
                                        resume_text=st.session_state.resume_text,
                                    )
                                    if "error" in updated:
                                        st.error(updated["error"])
                                    else:
                                        # Update all session state variables
                                        st.session_state.posts[idx]["topic_title"] = updated["topic_title"]
                                        st.session_state.posts[idx]["content"] = updated["content"]
                                        st.session_state.posts[idx]["word_count"] = updated["word_count"]
                                        st.session_state.generated_contents[idx] = updated["content"]
                                        st.session_state.revision_version[idx] = revision_ver + 1
                                        st.success(f"✅ Updated to: {updated['topic_title']}")
                                        st.rerun()
                            else:
                                st.warning("Please enter feedback")
                
                with col_b2:
                    # Post to LinkedIn
                    if st.button("🔗 Post to LinkedIn", key=f"linkedin_{idx}", use_container_width=True):
                        open_linkedin_with_text(st.session_state.posts[idx]["content"])
                
                with col_b3:
                    # Schedule this post
                    if st.button("⏰ Schedule", key=f"schedule_{idx}", use_container_width=True):
                        schedule_date = st.session_state.generator.create_schedule(
                            start_date=datetime.now().strftime("%Y-%m-%d"),
                            post_times=st.session_state.post_times
                        )
                        if schedule_date and idx < len(schedule_date):
                            scheduled = st.session_state.generator.scheduler.schedule_post(
                                post_content=st.session_state.posts[idx]["content"],
                                post_time=schedule_date[idx]["time"],
                                post_date=schedule_date[idx]["date"]
                            )
                            if scheduled:
                                st.success(f"✅ Scheduled for {schedule_date[idx]['date']} at {schedule_date[idx]['time']}")
                
                st.markdown('</div>', unsafe_allow_html=True)

# ============ TAB 2: SCHEDULE ============
with tab_schedule:
    if not st.session_state.posts:
        st.info("📝 Generate posts first to create a schedule.")
    else:
        col_s1, col_s2 = st.columns([2, 1])
        
        with col_s1:
            start_date = st.date_input("📅 Start Date", datetime.today())
            
            if st.button("📋 Create Schedule", type="primary", use_container_width=True):
                schedule_list = st.session_state.generator.create_schedule(
                    start_date.strftime("%Y-%m-%d"),
                    post_times=st.session_state.post_times
                )
                st.session_state.schedule = schedule_list
                st.success(f"✅ Created schedule for {len(schedule_list)} posts!")
                
                # Auto-schedule if enabled
                if st.session_state.auto_schedule:
                    scheduled = st.session_state.generator.auto_schedule_posts(
                        start_date=start_date.strftime("%Y-%m-%d"),
                        post_times=st.session_state.post_times
                    )
                    if scheduled:
                        st.info(f"🤖 Auto-scheduled {len(schedule_list)} posts for posting!")
        
        with col_s2:
            st.markdown("### ⏰ Posting Times")
            for i, time in enumerate(st.session_state.post_times):
                st.write(f"{i+1}. {time}")
        
        if st.session_state.schedule:
            st.markdown("---")
            st.markdown("### 📅 Content Calendar")
            
            # Convert to DataFrame for better display
            df = pd.DataFrame(st.session_state.schedule)
            
            # Add status column
            df['Status'] = df.apply(lambda x: 
                '✅ Ready' if st.session_state.auto_schedule else '⏳ Pending', axis=1)
            
            # Display as table
            st.dataframe(
                df[['day', 'date', 'day_of_week', 'time', 'topic', 'Status']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "day": "Day",
                    "date": "Date",
                    "day_of_week": "Day",
                    "time": "Time",
                    "topic": "Topic",
                    "Status": st.column_config.TextColumn("Status")
                }
            )
            
            # Calendar view
            st.markdown("### 📆 Calendar View")
            for row in st.session_state.schedule:
                col_c1, col_c2, col_c3 = st.columns([1, 3, 2])
                with col_c1:
                    st.markdown(f"**Day {row['day']}**")
                with col_c2:
                    st.markdown(f"{row['date']} ({row['day_of_week']}) • {row['time']}")
                with col_c3:
                    # Safely get topic preview
                    topic_preview = row['topic'][:30] + "..." if len(row['topic']) > 30 else row['topic']
                    st.markdown(f"`{topic_preview}`")
                st.markdown("---")

# ============ TAB 3: ANALYTICS ============
with tab_stats:
    if not st.session_state.posts:
        st.info("📊 Generate posts first to see analytics.")
    else:
        # Statistics
        stats = st.session_state.generator.get_statistics()
        
        # Metrics in cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Total Posts</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(stats.get("total_posts", 0)), unsafe_allow_html=True)
        
        with col_m2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Total Words</div>
                <div class="metric-value">{:,}</div>
            </div>
            """.format(stats.get("total_words", 0)), unsafe_allow_html=True)
        
        with col_m3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Avg Words</div>
                <div class="metric-value">{:.0f}</div>
            </div>
            """.format(stats.get("avg_words_per_post", 0)), unsafe_allow_html=True)
        
        with col_m4:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Word Limit</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(stats.get("word_limit_compliance", "0%")), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Detailed statistics
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("### 📈 Post Length Distribution")
            word_counts = [p["word_count"] for p in st.session_state.posts]
            df_words = pd.DataFrame({
                "Post": [f"Day {i+1}" for i in range(len(word_counts))],
                "Words": word_counts,
                "Target": st.session_state.max_words
            })
            st.bar_chart(df_words.set_index("Post")[["Words", "Target"]])
        
        with col_s2:
            st.markdown("### ⭐ Quality Scores")
            quality_scores = [p.get("quality_score", 0) for p in st.session_state.posts]
            df_quality = pd.DataFrame({
                "Post": [f"Day {i+1}" for i in range(len(quality_scores))],
                "Score": quality_scores
            })
            st.line_chart(df_quality.set_index("Post"))
        
        # Export options
        st.markdown("---")
        st.markdown("### 📤 Export Posts")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            json_data = st.session_state.generator.export_posts("json")
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"linkedin_posts_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_e2:
            csv_data = st.session_state.generator.export_posts("csv")
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"linkedin_posts_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_e3:
            # Export schedule
            if st.session_state.schedule:
                schedule_df = pd.DataFrame(st.session_state.schedule)
                schedule_csv = schedule_df.to_csv(index=False)
                st.download_button(
                    label="📅 Download Schedule",
                    data=schedule_csv,
                    file_name=f"post_schedule_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ============ TAB 4: AUTOMATION ============
with tab_automation:
    st.markdown("### 🤖 Auto-Posting Automation")
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        <div class="professional-card">
            <h4>🚀 Auto-Scheduler Status</h4>
            <p>Automatically post content to LinkedIn at scheduled times.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Scheduler status
        scheduled_posts = st.session_state.generator.scheduler.get_scheduled_posts()
        
        if scheduled_posts:
            st.success(f"✅ {len(scheduled_posts)} posts scheduled for auto-posting")
            
            st.markdown("#### 📋 Scheduled Posts:")
            for i, post in enumerate(scheduled_posts[:5]):  # Show first 5
                # Safely get content preview with fallback
                content_preview = post.get('content_preview', '')
                if not content_preview and 'content' in post:
                    content_preview = post['content'][:100] + "..."
                elif not content_preview:
                    content_preview = "No preview available"
                
                scheduled_time = post.get('scheduled_time', 'N/A')
                status = post.get('status', 'unknown')
                
                st.markdown(f"""
                **{i+1}. {scheduled_time}**
                - Preview: {content_preview}
                - Status: `{status}`
                """)
            
            if len(scheduled_posts) > 5:
                st.info(f"... and {len(scheduled_posts) - 5} more posts")
        else:
            st.info("No posts scheduled yet. Enable auto-scheduling in settings.")
    
    with col_a2:
        st.markdown("""
        <div class="professional-card">
            <h4>⚙️ Automation Settings</h4>
            <p>Configure how and when posts are automated.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Automation controls
        auto_enabled = st.toggle(
            "Enable Auto-Posting",
            value=st.session_state.auto_schedule,
            key="auto_toggle"
        )
        st.session_state.auto_schedule = auto_enabled
        
        if auto_enabled:
            st.success("🤖 Auto-posting enabled!")
            
            # Post frequency
            frequency = st.selectbox(
                "Posting Frequency",
                ["Once daily", "Twice daily", "Every other day", "Custom"],
                index=0
            )
            
            # Time slots
            st.markdown("**Preferred Posting Times:**")
            for i, time in enumerate(st.session_state.post_times):
                st.write(f"{i+1}. {time}")
            
            # Auto-schedule button
            if st.button("🔄 Schedule All Posts", use_container_width=True):
                if st.session_state.posts:
                    scheduled = st.session_state.generator.auto_schedule_posts(
                        start_date=datetime.now().strftime("%Y-%m-%d"),
                        post_times=st.session_state.post_times
                    )
                    if scheduled:
                        st.success("✅ All posts scheduled for auto-posting!")
                        st.rerun()
                else:
                    st.warning("Generate posts first!")
        else:
            st.warning("Auto-posting is disabled")

# ============ END OF FILE ============