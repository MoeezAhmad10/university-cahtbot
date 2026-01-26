import streamlit as st
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import time

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Riphah ChatBot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stMainBlockContainer {
        padding-top: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

class RiphahChatbot:
    SYSTEM_PROMPT = """You are an expert assistant for Riphah International University. Your role is to provide accurate, helpful information about Riphah's programs, campuses, admissions, and facilities. GUIDELINES: 1. Answer based ONLY on the provided context about Riphah University 2. Be specific and cite relevant details (programs, campus names, admission requirements) 3. For fee/tuition queries: Provide detailed breakdown if available 4. Use a professional yet friendly tone 5. Keep responses concise unless asked for details 6. If asked about multiple campuses, distinguish clearly: Islamabad, Lahore, Faisalabad, Sahiwal IMPORTANT LINKS: - Main Website: https://riphah.edu.pk/ - All Programs: https://riphah.edu.pk/programs/ - Fee Structure: https://riphah.edu.pk/fee-structure/ - Admissions: https://riphah.edu.pk/admissions/ - Scholarships: https://riphah.edu.pk/scholarships/ - Contact: https://riphah.edu.pk/contact/"""

    def __init__(self, openai_api_key, cache_dir="./chatbot_cache"):
        self.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.7,
            max_tokens=300
        )
        self.vectorstore = None
        self.retriever = None
        self.prompt = None
        self.riphah_urls = [
            "https://riphah.edu.pk/",
            "https://riphah.edu.pk/about/",
            "https://riphah.edu.pk/university-introduction-2/",
            "https://riphah.edu.pk/about/vision-mission-values/",
            "https://riphah.edu.pk/academics/",
            "https://riphah.edu.pk/academics/faculties/",
            "https://riphah.edu.pk/programs/",
            "https://riphah.edu.pk/admissions/",
            "https://riphah.edu.pk/fee-structure/",
            "https://riphah.edu.pk/tuition-fees/",
            "https://riphah.edu.pk/fhms/",
            "https://riphah.edu.pk/fhms/admission/",
            "https://riphah.edu.pk/islamabad-campus/",
            "https://riphah.edu.pk/lahore-campus/",
            "https://riphahfsd.edu.pk/",
            "https://riphahfsd.edu.pk/admissions/",
            "https://riphahfsd.edu.pk/programs/",
            "https://riphahsahiwal.edu.pk/",
            "https://riphahsahiwal.edu.pk/admissions/",
            "https://riphah.edu.pk/research/",
            "https://riphah.edu.pk/oric/",
            "https://riphah.edu.pk/innovation-hub/",
            "https://riphah.edu.pk/about/hospitals/",
            "https://riphah.edu.pk/campus-life/",
            "https://riphah.edu.pk/student-services/",
            "https://riphah.edu.pk/scholarships/",
            "https://riphah.edu.pk/financial-aid/",
            "https://riphah.edu.pk/hostel/",
            "https://riphah.edu.pk/library/",
            "https://riphah.edu.pk/faculty/",
            "https://riphah.edu.pk/careers/",
            "https://riphah.edu.pk/contact/",
            "https://riphah.edu.pk/contact-us/",
            "https://riphah.edu.pk/administration/",
            "https://riphah.edu.pk/offices/",
            "https://riphah.edu.pk/news/",
            "https://riphah.edu.pk/events/",
            "https://riphah.edu.pk/quality-assurance/",
            "https://riphah.edu.pk/accreditation/",
            "https://riphah.edu.pk/online-programs/",
            "https://riphah.edu.pk/distance-education/",
            "https://riphahfsd.edu.pk/fee-structure/",
            "https://riphahfsd.edu.pk/campus-life/",
            "https://riphahfsd.edu.pk/contact/",
            "https://riphahsahiwal.edu.pk/fee-structure/",
            "https://riphahsahiwal.edu.pk/programs/",
            "https://riphahsahiwal.edu.pk/contact/",
        ]

    def _get_cache_path(self, cache_type="vectorstore"):
        return self.cache_dir / f"riphah_{cache_type}_{datetime.now().strftime('%Y%m%d')}"

    def _load_cached_vectorstore(self):
        cache_path = self._get_cache_path("vectorstore")
        if cache_path.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(cache_path), self.embeddings, allow_dangerous_deserialization=True
                )
                return True
            except:
                return False
        return False

    def _save_vectorstore(self):
        try:
            cache_path = self._get_cache_path("vectorstore")
            self.vectorstore.save_local(str(cache_path))
        except:
            pass

    def load_local_documents(self, doc_folder="./documents"):
        """Load documents from local folder (PDF, TXT, DOCX) - ANY filename works!"""
        all_documents = []
        doc_path = Path(doc_folder)
        
        if not doc_path.exists():
            st.warning(f"📁 Documents folder not found at {doc_folder}")
            st.info("✅ Create a 'documents' folder in your project and add your files (any name works!)")
            st.info("📄 Supported formats: PDF, TXT, DOCX")
            return all_documents
        
        # Load ANY file with these extensions (names don't matter)
        files = list(doc_path.glob("*.pdf")) + list(doc_path.glob("*.txt")) + list(doc_path.glob("*.docx"))
        
        if not files:
            st.warning(f"❌ No documents found in {doc_folder}")
            st.info("📄 Add PDF, TXT, or DOCX files to the documents folder")
            return all_documents
        
        st.success(f"✅ Found {len(files)} document(s) to load")
        progress_bar = st.progress(0)
        status_text = st.empty()
        loaded_count = 0
        
        for i, file_path in enumerate(files):
            status_text.text(f"📖 Loading: {file_path.name}")
            try:
                if file_path.suffix.lower() == ".pdf":
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                elif file_path.suffix.lower() == ".txt":
                    from langchain_community.document_loaders import TextLoader
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    documents = loader.load()
                elif file_path.suffix.lower() == ".docx":
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(str(file_path))
                    documents = loader.load()
                else:
                    continue
                
                if documents:
                    # Add filename metadata to track source
                    for doc in documents:
                        doc.metadata['source_file'] = file_path.name
                    all_documents.extend(documents)
                    loaded_count += 1
                    st.success(f"✅ Loaded {len(documents)} chunks from {file_path.name}")
            except Exception as e:
                st.error(f"❌ Failed to load {file_path.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(files))
        
        progress_bar.empty()
        status_text.empty()
        
        if loaded_count > 0:
            st.success(f"✅ Successfully loaded {loaded_count} document(s) with {len(all_documents)} chunks!")
        
        return all_documents

    def load_specific_urls(self, urls=None):
        """Load documents from URLs"""
        if urls is None:
            urls = self.riphah_urls
        all_documents = []
        successful = 0
        failed = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        for i, url in enumerate(urls):
            status_text.text(f"Loading: {url}")
            try:
                loader = WebBaseLoader(url)
                documents = loader.load()
                if documents and len(documents[0].page_content) > 100:
                    all_documents.extend(documents)
                    successful += 1
                else:
                    failed += 1
                time.sleep(0.3)
            except:
                failed += 1
            progress_bar.progress((i + 1) / len(urls))
        progress_bar.empty()
        status_text.empty()
        return all_documents

    def process_documents(self, documents):
        if not documents:
            return False
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        self._save_vectorstore()
        self._setup_retriever_and_prompt()
        return True

    def _setup_retriever_and_prompt(self):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        template = f"""{self.SYSTEM_PROMPT}

Context: {{context}}

Question: {{question}}

Answer:"""
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def initialize(self, use_cache=True, use_local=True):
        if use_cache and self._load_cached_vectorstore():
            self._setup_retriever_and_prompt()
            return True
        
        # Try loading from local documents first
        if use_local:
            documents = self.load_local_documents("./documents")
        
        # If no local documents, fall back to URLs
        if not documents:
            st.info("Loading from website URLs...")
            documents = self.load_specific_urls()
        
        return self.process_documents(documents)

    def chat(self, question):
        if self.retriever is None or self.prompt is None:
            return "Please initialize the chatbot first", []
        try:
            # Get RAG response with university-specific data
            docs = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            rag_prompt = f"""{self.SYSTEM_PROMPT}

UNIVERSITY DATABASE CONTEXT:
{context}

User Question: {question}

Based ONLY on the university database above, provide a specific and accurate answer. Include faculty names, program details, and exact information from the database."""
            
            rag_response = self.llm.invoke(rag_prompt)
            
            # Get enhanced real-time response
            realtime_prompt = f"""You are Riphah International University's expert assistant. Your role is to provide COMPREHENSIVE, DETAILED, and ACCURATE information.

Key responsibilities:
1. Provide SPECIFIC names of faculty members when asked
2. Give EXACT program details and requirements
3. Include PRECISE fee structures and financial information
4. Mention specific campus locations and facilities
5. Provide detailed admission procedures and deadlines
6. Discuss research opportunities and achievements
7. Highlight student services and support systems
8. Be specific about department functions and offerings

Question: {question}

Provide a DETAILED response with specific facts, names, numbers, and examples. Do NOT be vague. Include:
- Exact names when discussing people
- Specific amounts for fees/costs
- Precise details about programs
- Clear procedures and steps
- Department and campus specifics"""
            
            realtime_response = self.llm.invoke(realtime_prompt)
            
            # INTELLIGENT HYBRID SYNTHESIS
            hybrid_prompt = f"""You are an EXPERT information synthesizer for Riphah International University. Your task is to create the ULTIMATE BEST response by intelligently combining two sources of information.

SOURCE 1 - OFFICIAL DATABASE (Most Important - University Official Records):
{rag_response.content}

SOURCE 2 - KNOWLEDGE ENHANCEMENT (Supporting Context):
{realtime_response.content}

USER QUESTION: {question}

YOUR TASK - CREATE THE BEST POSSIBLE RESPONSE:
1. **PRIORITIZE Official Data**: Use facts from SOURCE 1 (database) as the foundation
2. **ENHANCE with Knowledge**: Add supporting details from SOURCE 2 where they enhance clarity
3. **INCLUDE SPECIFIC DETAILS**:
   - Faculty names and titles (from database)
   - Program codes and exact requirements
   - Fee amounts and payment schedules
   - Campus locations and facilities
   - Department heads and contacts
   - Exact procedures with step-by-step instructions
4. **CROSS-REFERENCE**: If both sources mention something, use the specific version
5. **STRUCTURE CLEARLY**: Use bullet points for lists, clear paragraphs for explanations
6. **BE COMPREHENSIVE**: Cover all aspects of the question
7. **CITE SOURCES**: Mention which information comes from official records

GUIDELINES FOR RESPONSE:
- If database has specific info, USE IT and don't paraphrase
- Add context from general knowledge to explain "WHY"
- Be specific about names, numbers, and procedures
- Format nicely with headers and bullet points
- Include relevant links or contact information if known
- Never contradict official university data
- If uncertain about something, say "Please verify with [department/office]"

CREATE THE BEST, MOST HELPFUL RESPONSE NOW:"""
            
            final_response = self.llm.invoke(hybrid_prompt)
            sources = list(set([doc.metadata.get('source_file', doc.metadata.get('source', 'Database')) for doc in docs]))
            
            return final_response.content, sources
        except Exception as e:
            return f"Error processing your question: {str(e)[:100]}", []

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.messages = []
    st.session_state.initialized = False
    st.session_state.current_page = "home"

# TOP NAVIGATION BAR
st.markdown("""
<div style="background: linear-gradient(to right, #1a2540 0%, #2c5aa0 70%); padding: 10px 20px; border-bottom: 2px solid #2c3e50; display: flex; justify-content: space-between; align-items: center;">
    <div></div>
    <div style="display: flex; gap: 30px; align-items: center;">
        <a href="#" style="color: #f39c12; text-decoration: none; font-size: 12px; font-weight: 600; transition: color 0.3s;" onmouseover="this.style.color='#f1c40f'" onmouseout="this.style.color='#f39c12'">Apply Now</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Careers</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Moellim</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Contact</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Home</a>
    </div>
</div>
""", unsafe_allow_html=True)

# MAIN NAVIGATION BAR
st.markdown("""
<div style="background-color: #1a2540; padding: 12px 20px; border-bottom: 2px solid #2c3e50; display: flex; align-items: center; justify-content: space-between; gap: 30px;">
    <div style="display: flex; align-items: center; gap: 12px; flex-shrink: 0;">
        <div style="width: 45px; height: 45px; background: url('https://riphah.edu.pk/wp-content/uploads/2020/09/logo-1.png') center/contain no-repeat; filter: brightness(0) invert(1);"></div>
        <div style="color: white;">
            <div style="font-size: 13px; font-weight: 700; margin: 0; letter-spacing: 0.8px;">RIPHAH</div>
            <div style="font-size: 8px; color: #9ca3af; margin: 0; letter-spacing: 0.5px;">INTERNATIONAL UNIVERSITY</div>
        </div>
    </div>
    <div style="display: flex; gap: 20px; flex: 1; justify-content: center; align-items: center;">
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">About</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Academics</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Admissions</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Research</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Hospitals</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Campus Life</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">News</a>
        <a href="#" style="color: white; text-decoration: none; font-size: 12px; font-weight: 500; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">Offices</a>
    </div>
    <div style="display: flex; gap: 12px; align-items: center; flex-shrink: 0;">
        <a href="#" style="color: white; text-decoration: none; font-size: 18px; transition: color 0.3s; cursor: pointer;" onmouseover="this.style.color='#bfdbfe'" onmouseout="this.style.color='white'">🔍</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Chat button below navbar
st.markdown("""
<div style="background-color: #1a2540; padding: 0px 20px 15px 20px; display: flex; justify-content: flex-end;">
    <button onclick="document.querySelector('[data-testid=\"stForm\"]')?.scrollIntoView();" style="background-color: transparent; color: white; border: 1.5px solid white; padding: 8px 20px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 12px; transition: all 0.3s;" onmouseover="this.style.backgroundColor='rgba(255,255,255,0.1)'" onmouseout="this.style.backgroundColor='transparent'">💬 Chat</button>
</div>
""", unsafe_allow_html=True)

# Hidden chat button for functionality
col_chat_hidden = st.columns(1)[0]
with col_chat_hidden:
    st.markdown("<div style='display: none;'>", unsafe_allow_html=True)
    if st.button("Open Chat", key="nav_chat_btn_hidden"):
        st.session_state.current_page = "chat"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)



# Sidebar for Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key"
    )
    
    st.markdown("### 📚 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Local Documents", "Website URLs", "Both (Local First)"],
        help="Local Documents: Load from ./documents folder\nWebsite URLs: Load from Riphah website\nBoth: Try local first, then URLs"
    )
    
    if st.button("Initialize Chatbot", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key")
        else:
            with st.spinner("Initializing chatbot..."):
                try:
                    st.session_state.chatbot = RiphahChatbot(api_key)
                    use_local = data_source != "Website URLs"
                    if st.session_state.chatbot.initialize(use_local=use_local):
                        st.session_state.initialized = True
                        st.success("Chatbot initialized successfully!")
                    else:
                        st.error("Failed to initialize chatbot")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    if st.button("Refresh Data", use_container_width=True):
        if st.session_state.chatbot:
            with st.spinner("Refreshing data..."):
                use_local = data_source != "Website URLs"
                if st.session_state.chatbot.initialize(use_cache=False, use_local=use_local):
                    st.success("Data refreshed successfully!")
        else:
            st.warning("Initialize chatbot first")
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat history cleared!")
    
    st.markdown("---")
    st.markdown("### 📁 Setup Instructions")
    st.info("""
    **To use local documents:**
    1. Create a `documents` folder in your project
    2. Add your files:
       - Faculty list (PDF/TXT)
       - Programs (PDF/TXT)
       - Fee structure (PDF/TXT)
       - Any other relevant docs
    3. Select "Local Documents" or "Both"
    4. Click "Initialize Chatbot"
    
    **Supported formats:** PDF, TXT, DOCX
    """)

# Page Content
if st.session_state.current_page == "chat":
    if not st.session_state.initialized:
        st.error("❌ Please initialize the chatbot using the sidebar configuration first.")
    else:
        # Chat Header
        st.markdown("""
        <div style="background-color: #1e40af; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 24px;">💬 Riphah University Assistant</h2>
            <p style="margin: 5px 0 0 0; font-size: 13px; color: #bfdbfe;">Ask questions about admissions, programs, and more</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Messages Container
        messages_container = st.container()
        
        with messages_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align: center; padding: 60px 20px; color: #9ca3af;">
                    <h3 style="color: #d1d5db; margin-bottom: 10px;">Welcome to Riphah ChatBot</h3>
                    <p>Ask me anything about Riphah International University. I can help you with information about programs, admissions, campuses, and more.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Example Questions
                st.markdown("#### Try asking:")
                cols = st.columns(2)
                example_questions = [
                    "What programs does Riphah offer?",
                    "Tell me about MBBS admissions",
                    "What is the fee structure?",
                    "How can I apply for admission?"
                ]
                
                for idx, question in enumerate(example_questions):
                    with cols[idx % 2]:
                        if st.button(question, use_container_width=True, key=f"example_{idx}"):
                            st.session_state.messages.append({
                                "role": "user",
                                "content": question
                            })
                            with st.spinner("Thinking..."):
                                response, sources = st.session_state.chatbot.chat(question)
                                st.session_state.messages.append({
                                    "role": "bot",
                                    "content": response,
                                    "sources": sources
                                })
                            st.rerun()
            else:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div style="text-align: right; margin: 10px 0;">
                            <div style="display: inline-block; background-color: #1e40af; color: white; padding: 12px 16px; border-radius: 8px; max-width: 60%;">
                                {message['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="text-align: left; margin: 10px 0;">
                            <div style="display: inline-block; background-color: #374151; color: #e5e7eb; padding: 12px 16px; border-radius: 8px; max-width: 60%;">
                                {message['content']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if "sources" in message and message["sources"]:
                            with st.expander("📎 Sources", expanded=False):
                                for source in message["sources"]:
                                    st.write(f"[{source}]({source})")
        
        # Input Area
        st.markdown("<hr style='margin: 20px 0; border-color: #374151;'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Your question:",
                placeholder="Type your question here...",
                key="user_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_clicked = st.button("Send", use_container_width=True, type="primary")
        
        if send_clicked and user_input:
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            with st.spinner("Thinking..."):
                response, sources = st.session_state.chatbot.chat(user_input)
                st.session_state.messages.append({
                    "role": "bot",
                    "content": response,
                    "sources": sources
                })
            
            st.rerun()

else:
    # Home Page
    st.markdown("""
    <div style="background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 1200 600%22%3E%3Crect fill=%22%23111827%22 width=%221200%22 height=%22600%22/%3E%3C/svg%3E'); background-size: cover; background-position: center; height: 500px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: white; text-align: center; padding: 20px; border-radius: 8px;">
        <p style="font-size: 18px; color: #e5e7eb; margin: 0 0 20px 0;">WELCOME TO</p>
        <h1 style="font-size: 56px; font-weight: bold; margin: 20px 0;">Riphah International University</h1>
        <p style="font-size: 18px; color: #e5e7eb; margin: 10px 0;">Islamabad, Rawalpindi, Lahore, Malakand</p>
        <p style="font-size: 16px; color: #d1d5db;">Faisalabad, Sahiwal, Peshawar, Gujranwala</p>
    </div>
    """, unsafe_allow_html=True)