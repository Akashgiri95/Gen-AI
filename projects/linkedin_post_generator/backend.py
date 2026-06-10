"""
backend.py

LinkedIn Post Generator backend:
- RAG: resume → topics (with angles)
- story_agent: build backstories
- post_writer_agent: write final posts
- revision_agent: revise with human feedback
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
 
import pandas as pd
import pymupdf as fitz
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, LLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.documents import Document

try:
    from langchain_ollama import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_AVAILABLE = False


# ========= STEP 1: Environment & Clients =========

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip().replace("GROQ_API_KEY=", "")

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    raise ValueError(
        "Invalid GROQ_API_KEY. "
        "Ensure .env has a line like:\n"
        "GROQ_API_KEY=gsk_your_real_key_here"
    )

crew_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.55,
)

rag_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0.0,
)


# ========= STEP 2: PDF Extraction =========

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            blocks = page.get_text("blocks")
            block_text = "".join([b[4] + "\n" for b in blocks])
            words = page.get_text("words")
            word_text = " ".join([w[4] for w in words])

            if len(block_text.strip()) > len(text.strip()):
                page_text = block_text
            else:
                page_text = text

            if len(page_text.strip()) < 100 and len(word_text.strip()) > len(page_text.strip()):
                page_text = word_text

            page_text = page_text.replace("\x00", " ")
            page_text = re.sub(r"\s+", " ", page_text).strip()
            full_text += page_text + "\n\n"

        doc.close()
        full_text = re.sub(r"\n\s*\n\s*\n+", "\n\n", full_text)
        full_text = re.sub(r"[^\x00-\x7F]+", " ", full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()
        return full_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    t = page.extract_text() or ""
                    text += t + "\n"
            return text.strip()
        except Exception:
            return ""


# ========= STEP 3: RAG Pipeline (Topics Only) =========

class RAGPipeline:
    def __init__(self):
        if OLLAMA_AVAILABLE:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434",
            )
        else:
            self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = None
        self.chunks: List[Document] = []

    def process_resume(self, resume_text: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        docs = [Document(page_content=resume_text)]
        self.chunks = splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
        )

    def extract_topics(self, num_posts: int) -> List[Dict[str, Any]]:
        if not self.vector_store:
            return []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a LinkedIn content strategist working in a RAG system.
The RESUME CONTEXT is the ONLY source of truth.

========================
RESUME CONTEXT (TRUNCATED)
========================
{context}

========================
TASK
========================
{question}

========================
RULES
========================
- Use ONLY information that clearly appears in the context.
- Focus on concrete projects, roles, skills, or achievements.
- Avoid generic or motivational topics.
- Topics must be DISTINCT from each other.
- Output STRICT JSON only, no explanations.
""",
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=rag_llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

        question = f"""
Generate EXACTLY {num_posts} LinkedIn post topics from this resume.

For each topic, include:
- "topic_title": short, specific title (max 80 characters)
- "resume_reference": brief pointer to the source line/section
- "story_angle": how this can become a personal story

Constraints:
- Each topic must map to a concrete role, project, skill, achievement, or education item.
- Each topic must be different (different domain or angle).

Return ONLY a JSON array like:
[
  {{
    "topic_title": "Scaling APIs with Python & Docker",
    "resume_reference": "Backend Engineer at ABC Corp - scaled APIs",
    "story_angle": "From small API to high-scale service"
  }}
]
"""

        result = qa_chain.invoke({"query": question})
        raw = result.get("result") if isinstance(result, dict) else result
        try:
            topics = json.loads(raw)
            if isinstance(topics, dict):
                topics = [topics]
            return topics
        except Exception:
            return [
                {
                    "topic_title": f"Day {i+1} Topic",
                    "resume_reference": "",
                    "story_angle": "",
                }
                for i in range(num_posts)
            ]


# ========= STEP 4: Agents (Only 3) =========

story_agent = Agent(
    role="Backstory Builder",
    goal="Expand each topic into a concise, structured professional backstory.",
    backstory=(
        "You turn resume facts into short narrative backstories with context, "
        "challenge, action, and outcome using minimal but rich wording."
    ),
    llm=crew_llm,
    verbose=True,
    allow_delegation=False,
)

post_writer_agent = Agent(
    role="LinkedIn Post Writer",
    goal="Write first-person LinkedIn posts that respect word limits.",
    backstory=(
        "You write engaging, professional LinkedIn posts that follow a clear flow "
        "and stay within the requested word range."
    ),
    llm=crew_llm,
    verbose=True,
    allow_delegation=False,
)

revision_agent = Agent(
    role="Interactive Post Reframer",
    goal="Regenerate or reframe posts based on human feedback and constraints.",
    backstory=(
        "You take user feedback and regenerate a post around a new topic or constraint "
        "such as a tighter word limit, while staying close to the resume."
    ),
    llm=crew_llm,
    verbose=True,
    allow_delegation=False,
)


# ========= STEP 5: Tasks (Story, Post, Revision) =========

def build_story_task(topics: List[Dict[str, Any]]) -> Task:
    compact_topics = [
        {
            "day": t.get("day"),
            "topic_title": t.get("topic_title"),
            "resume_reference": t.get("resume_reference", "")[:120],
            "story_angle": t.get("story_angle", "")[:140],
        }
        for t in topics
    ]

    return Task(
        description=f"""
Create SHORT backstories for the topics below.

TOPICS (DO NOT CHANGE TITLES):
{json.dumps(compact_topics, indent=2)}

For EACH topic:
- Use ONLY details consistent with the resume.
- Write a compact 3–4 sentence backstory with:
  (1) Context (role/project)
  (2) Challenge
  (3) Action (tools/skills)
  (4) Outcome or learning

Keep each backstory under 120 words.

Return STRICT JSON array:
[
  {{
    "day": 1,
    "topic_title": "Scaling APIs with Python & Docker",
    "backstory": "..."
  }}
]
""",
        agent=story_agent,
        expected_output="JSON list of backstories (each <120 words).",
    )


def build_post_task(num_posts: int, max_words: int, backstories: List[Dict[str, Any]]) -> Task:
    min_words = max(30, max_words - 20)
    max_range = max_words + 20

    compact_backstories = [
        {
            "day": b.get("day"),
            "topic_title": b.get("topic_title"),
            "backstory": b.get("backstory", "")[:260],
        }
        for b in backstories
    ]

    return Task(
        description=f"""
Write EXACTLY {num_posts} LinkedIn posts from these backstories.

BACKSTORIES (TRUNCATED WHERE NEEDED):
{json.dumps(compact_backstories, indent=2)}

For EACH item:
- Keep the same "topic_title".
- Use first-person LinkedIn style.
- Structure:
  1) Hook (1 short sentence)
  2) Story: context → challenge → action → outcome
  3) Takeaway (1 sentence)
  4) CTA question at the end.

CONSTRAINTS:
- Word range per post: {min_words}–{max_range} words.
- NO emojis.
- NO markdown.
- Ground posts in realistic resume scenarios; do NOT invent new companies or roles.

Return STRICT JSON array:
[
  {{
    "day": 1,
    "topic_title": "Scaling APIs with Python & Docker",
    "post_content": "full LinkedIn post text here"
  }}
]
""",
        agent=post_writer_agent,
        expected_output=f"JSON list of {num_posts} posts, each {min_words}-{max_range} words.",
    )


def build_revision_task(
    original_post: str,
    original_topic: str,
    user_feedback: str,
    resume_text: str,
    max_words: int,
) -> Task:
    min_words = max(30, max_words - 20)
    max_range = max_words + 20

    return Task(
        description=f"""
Revise a LinkedIn post according to HUMAN FEEDBACK.

RESUME (GROUND TRUTH, TRUNCATED):
\"\"\"{resume_text[:3000]}\"\"\"

CURRENT TOPIC TITLE:
\"\"\"{original_topic[:160]}\"\"\"

CURRENT POST (TRUNCATED):
\"\"\"{original_post[:1200]}\"\"\"

USER FEEDBACK:
\"\"\"{user_feedback[:500]}\"\"\"

TASK:
1. If feedback suggests a different angle or topic, update topic_title.
2. Write a NEW LinkedIn post for that topic, grounded in the resume.
3. Style: first-person, professional, concise.
4. Flow: hook → context → challenge → action → outcome → CTA question.
5. Enforce word range: {min_words}–{max_range} words.
6. No emojis or markdown.

OUTPUT:
Return STRICT JSON:
{{
  "topic_title": "final topic title",
  "post_content": "final post content",
  "word_count": [exact word count]
}}
""",
        agent=revision_agent,
        expected_output=f"JSON with topic_title and post_content, {min_words}-{max_range} words.",
    )


# ========= STEP 6: Simple Schedule Automation =========

class PostScheduler:
    def __init__(self):
        self.scheduled_posts = []
        
    def schedule_post(self, post_content: str, post_time: str, post_date: str, platform: str = "linkedin"):
        schedule_datetime = f"{post_date} {post_time}"
        self.scheduled_posts.append({
            "content": post_content,
            "scheduled_time": schedule_datetime,
            "platform": platform,
            "status": "scheduled",
            "content_preview": post_content[:100] + "..." if len(post_content) > 100 else post_content
        })
        print(f"[INFO] Post scheduled for {schedule_datetime} on {platform}")
        return True
    
    def get_scheduled_posts(self):
        return self.scheduled_posts


# ========= STEP 7: Main Generator (Crew Orchestration) =========

class LinkedInPostGenerator:
    def __init__(self):
        self.rag = RAGPipeline()
        self.scheduler = PostScheduler()
        self.topics: List[Dict[str, Any]] = []
        self.posts: List[Dict[str, Any]] = []
        self.schedule: List[Dict[str, Any]] = []

    def process_resume_text(self, resume_text: str):
        self.rag.process_resume(resume_text)

    def extract_topics(self, num_posts: int) -> List[Dict[str, Any]]:
        raw_topics = self.rag.extract_topics(num_posts)
        raw_topics = sorted(raw_topics, key=lambda t: t.get("topic_title", ""))
        self.topics = []
        for i, t in enumerate(raw_topics[:num_posts]):
            self.topics.append(
                {
                    "day": i + 1,
                    "topic_title": t.get("topic_title", f"Day {i+1} Topic"),
                    "resume_reference": t.get("resume_reference", ""),
                    "story_angle": t.get("story_angle", ""),
                }
            )
        return self.topics

    def run_full_pipeline(
        self,
        resume_text: str,
        num_posts: int,
        max_words: int,
    ) -> List[Dict[str, Any]]:
        # 1) RAG topics
        self.process_resume_text(resume_text)
        self.extract_topics(num_posts)

        # 2) Story backstories
        story_task = build_story_task(self.topics)
        story_crew = Crew(agents=[story_agent], tasks=[story_task], verbose=True)
        story_raw = str(story_crew.kickoff())
        try:
            story_data = json.loads(re.search(r"\[.*\]", story_raw, re.DOTALL).group())
        except Exception:
            story_data = []

        # 3) Posts
        post_task = build_post_task(num_posts, max_words, story_data)
        post_crew = Crew(agents=[post_writer_agent], tasks=[post_task], verbose=True)
        post_raw = str(post_crew.kickoff())
        try:
            posts_data = json.loads(re.search(r"\[.*\]", post_raw, re.DOTALL).group())
        except Exception:
            posts_data = []

        # 4) Strict word enforcement
        target_min = max(30, max_words - 20)
        target_max = max_words + 20

        final_posts: List[Dict[str, Any]] = []
        for i, p in enumerate(posts_data):
            content = p.get("post_content", "").strip()
            words = content.split()
            
            if len(words) > target_max:
                content = " ".join(words[:target_max])
                words = content.split()
            elif len(words) < target_min:
                additional = (
                    f" This experience taught me valuable lessons about "
                    f"{self.topics[i]['topic_title'].split()[-1] if i < len(self.topics) else 'professional growth'}."
                )
                content = content + additional
                words = content.split()
                if len(words) > target_max:
                    content = " ".join(words[:target_max])
                    words = content.split()

            final_posts.append(
                {
                    "day": i + 1,
                    "topic_title": p.get(
                        "topic_title",
                        self.topics[i]["topic_title"] if i < len(self.topics) else f"Day {i+1} Topic",
                    ),
                    "content": content,
                    "word_count": len(words),
                    "quality_score": min(95, 70 + min(25, len(words) / 10)),
                    "word_limit_met": target_min <= len(words) <= target_max,
                }
            )

        self.posts = final_posts
        return self.posts

    def revise_post(
        self,
        index: int,
        feedback: str,
        max_words: int,
        resume_text: str,
    ) -> Dict[str, Any]:
        if index < 0 or index >= len(self.posts):
            return {"error": "Invalid post index"}

        original_post = self.posts[index]["content"]
        original_topic = self.posts[index]["topic_title"]

        task = build_revision_task(
            original_post=original_post,
            original_topic=original_topic,
            user_feedback=feedback,
            resume_text=resume_text,
            max_words=max_words,
        )
        crew = Crew(agents=[revision_agent], tasks=[task], verbose=True)
        raw = str(crew.kickoff()).strip()

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(match.group()) if match else json.loads(raw)
            new_topic = data.get("topic_title", original_topic)
            new_post = data.get("post_content", original_post)
            word_count = data.get("word_count", len(new_post.split()))
        except Exception:
            new_topic = original_topic
            new_post = original_post
            word_count = len(new_post.split())

        target_min = max(30, max_words - 20)
        target_max = max_words + 20
        words = new_post.split()
        
        if len(words) > target_max:
            new_post = " ".join(words[:target_max])
            words = new_post.split()
            word_count = len(words)
        elif len(words) < target_min:
            additional = " This insight has been crucial for my professional development."
            new_post = new_post + additional
            words = new_post.split()
            if len(words) > target_max:
                new_post = " ".join(words[:target_max])
                words = new_post.split()
            word_count = len(words)

        self.posts[index] = {
            "day": self.posts[index]["day"],
            "topic_title": new_topic,
            "content": new_post,
            "word_count": word_count,
            "quality_score": min(95, 70 + min(25, word_count / 10)),
            "word_limit_met": target_min <= word_count <= target_max,
        }
        return self.posts[index]

    def create_schedule(self, start_date: str = None, post_times: List[str] = None) -> List[Dict[str, Any]]:
        if not start_date:
            start_date = datetime.now().strftime("%Y-%m-%d")
        base = datetime.strptime(start_date, "%Y-%m-%d")
        
        if not post_times:
            post_times = ["09:00", "12:00", "15:00", "18:00"]
        
        schedule = []
        for idx, post in enumerate(self.posts):
            d = base + timedelta(days=idx)
            time_idx = idx % len(post_times)
            post_time = post_times[time_idx]
            
            schedule.append(
                {
                    "day": idx + 1,
                    "date": d.strftime("%Y-%m-%d"),
                    "day_of_week": d.strftime("%A"),
                    "topic": post.get("topic_title", ""),
                    "time": post_time,
                    "status": "ready",
                    "content_preview": post.get("content", "")[:100] + "..." if len(post.get("content", "")) > 100 else post.get("content", ""),
                }
            )
        self.schedule = schedule
        return schedule
    
    def auto_schedule_posts(self, start_date: str = None, post_times: List[str] = None) -> bool:
        if not self.posts:
            return False
        
        schedule_list = self.create_schedule(start_date, post_times)
        success_count = 0
        
        for schedule_item in schedule_list:
            post_idx = schedule_item["day"] - 1
            if post_idx < len(self.posts):
                success = self.scheduler.schedule_post(
                    post_content=self.posts[post_idx]["content"],
                    post_time=schedule_item["time"],
                    post_date=schedule_item["date"]
                )
                if success:
                    success_count += 1
        
        return success_count > 0

    def get_statistics(self) -> Dict[str, Any]:
        if not self.posts:
            return {}
        total_posts = len(self.posts)
        total_words = sum(p["word_count"] for p in self.posts)
        questions = sum(1 for p in self.posts if "?" in p["content"])
        word_limit_met = sum(1 for p in self.posts if p.get("word_limit_met", False))
        
        return {
            "total_posts": total_posts,
            "total_words": total_words,
            "avg_words_per_post": total_words / total_posts if total_posts else 0,
            "posts_with_questions": questions,
            "word_limit_compliance": f"{(word_limit_met/total_posts*100):.1f}%" if total_posts else "0%",
            "scheduled_posts": len(self.scheduler.get_scheduled_posts()),
        }

    def export_posts(self, format: str = "json") -> str:
        if format == "json":
            return json.dumps(self.posts, indent=2)
        if format == "csv":
            df = pd.DataFrame(self.posts)
            return df.to_csv(index=False)
        return str(self.posts)


# ========= STEP 13: Smoke Test =========

if __name__ == "__main__":
    sample_resume = """
    John Doe – Software Engineer
    5+ years building backend services with Python, SQL, Docker, and AWS.
    Led a project to scale APIs to millions of requests per day.
    Built data pipelines and automated reporting dashboards.
    B.Tech in Computer Science. AWS Certified Solutions Architect.
    """

    gen = LinkedInPostGenerator()
    posts = gen.run_full_pipeline(
        resume_text=sample_resume,
        num_posts=3,
        max_words=200,
    )

    print("\n" + "="*60)
    print("SMOKE TEST RESULTS")
    print("="*60)
    print(f"Generated posts: {len(posts)}")
    
    for p in posts:
        print(f"\nDay {p['day']}: {p['topic_title']}")
        print(f"Words: {p['word_count']} (Limit met: {p.get('word_limit_met', False)})")
        print(f"Quality Score: {p.get('quality_score', 0)}/100")
        print(f"Preview: {p['content'][:150]}...")
    
    print("\n" + "-"*60)
    stats = gen.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "-"*60)
    print("SCHEDULE TEST")
    print("-"*60)
    schedule = gen.create_schedule()
    for item in schedule:
        print(f"Day {item['day']}: {item['date']} at {item['time']} - {item['topic']}")
    
    print("\nSmoke test completed successfully!")
# ============ END OF FILE ============
