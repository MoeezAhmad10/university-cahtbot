import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import time

# Load environment variables from .env file
load_dotenv()

class RiphahChatbot:
    # System prompt that guides the LLM behavior
    SYSTEM_PROMPT = """You are an expert assistant for Riphah International University. 
Your role is to provide accurate, helpful information about Riphah's programs, campuses, admissions, and facilities.

GUIDELINES:
1. Answer based ONLY on the provided context about Riphah University
2. Be specific and cite relevant details (programs, campus names, admission requirements)
3. For fee/tuition queries: Provide detailed breakdown if available, mention campus-specific fees
4. For admission queries: Always mention the specific campus and required documents
5. For program queries: Include program name, campus location, duration, and key highlights
6. Use a professional yet friendly tone
7. Keep responses concise unless asked for detailed information
8. Provide actionable next steps (contact info, website links) when relevant
9. If asked about multiple campuses, distinguish clearly: Islamabad, Lahore, Faisalabad, Sahiwal
10. When information is not in context, direct users to main website or contact office

IMPORTANT - ALWAYS REFERENCE THESE MAIN LINKS:
- Main Website: https://riphah.edu.pk/
- All Programs: https://riphah.edu.pk/programs/
- Fee Structure: https://riphah.edu.pk/fee-structure/
- Admissions: https://riphah.edu.pk/admissions/
- Scholarships: https://riphah.edu.pk/scholarships/
- Contact: https://riphah.edu.pk/contact/

For campus-specific information:
- Faisalabad Campus: https://riphahfsd.edu.pk/
- Sahiwal Campus: https://riphahsahiwal.edu.pk/

FORMAT: 
1. Start with direct answer
2. Provide supporting details from context
3. End with relevant link: "For more information, visit https://riphah.edu.pk/programs/ or contact admissions"
4. Always direct users to https://riphah.edu.pk/ as the main authority source"""

    def __init__(self, openai_api_key, cache_dir="./chatbot_cache"):
        """Initialize chatbot with API credentials and configuration"""
        self.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize embeddings model for converting text to vectors
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key
        )
        
        # Initialize language model for generating responses
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=openai_api_key,
            temperature=0.7,
            max_tokens=300
        )
        
        # Initialize components
        self.vectorstore = None
        self.retriever = None
        self.prompt = None
        
        # List of Riphah University URLs to scrape for information
        self.riphah_urls = [
            # Main pages
            "https://riphah.edu.pk/",
            "https://riphah.edu.pk/about/",
            "https://riphah.edu.pk/university-introduction-2/",
            "https://riphah.edu.pk/about/vision-mission-values/",
            
            # Academics
            "https://riphah.edu.pk/academics/",
            "https://riphah.edu.pk/academics/faculties/",
            "https://riphah.edu.pk/programs/",
            
            # Admissions & Fee Structure
            "https://riphah.edu.pk/admissions/",
            "https://riphah.edu.pk/fee-structure/",
            "https://riphah.edu.pk/tuition-fees/",
            "https://riphah.edu.pk/fhms/",
            "https://riphah.edu.pk/fhms/admission/",
            
            # Campuses
            "https://riphah.edu.pk/islamabad-campus/",
            "https://riphah.edu.pk/lahore-campus/",
            "https://riphahfsd.edu.pk/",
            "https://riphahfsd.edu.pk/admissions/",
            "https://riphahfsd.edu.pk/programs/",
            "https://riphahsahiwal.edu.pk/",
            "https://riphahsahiwal.edu.pk/admissions/",
            
            # Research & Innovation
            "https://riphah.edu.pk/research/",
            "https://riphah.edu.pk/oric/",
            "https://riphah.edu.pk/innovation-hub/",
            
            # Student Services
            "https://riphah.edu.pk/about/hospitals/",
            "https://riphah.edu.pk/campus-life/",
            "https://riphah.edu.pk/student-services/",
            "https://riphah.edu.pk/scholarships/",
            "https://riphah.edu.pk/financial-aid/",
            "https://riphah.edu.pk/hostel/",
            "https://riphah.edu.pk/library/",
            
            # Faculty & Staff
            "https://riphah.edu.pk/faculty/",
            "https://riphah.edu.pk/careers/",
            
            # Administrative
            "https://riphah.edu.pk/contact/",
            "https://riphah.edu.pk/contact-us/",
            "https://riphah.edu.pk/administration/",
            "https://riphah.edu.pk/offices/",
            
            # News & Events
            "https://riphah.edu.pk/news/",
            "https://riphah.edu.pk/events/",
            
            # Quality Assurance
            "https://riphah.edu.pk/quality-assurance/",
            "https://riphah.edu.pk/accreditation/",
            
            # Online & Distance Education
            "https://riphah.edu.pk/online-programs/",
            "https://riphah.edu.pk/distance-education/",
            
            # Faisalabad Campus Additional
            "https://riphahfsd.edu.pk/fee-structure/",
            "https://riphahfsd.edu.pk/campus-life/",
            "https://riphahfsd.edu.pk/contact/",
            
            # Sahiwal Campus Additional
            "https://riphahsahiwal.edu.pk/fee-structure/",
            "https://riphahsahiwal.edu.pk/programs/",
            "https://riphahsahiwal.edu.pk/contact/",
        ]

    def _get_cache_path(self, cache_type="vectorstore"):
        """Generate cache file path with current date"""
        return self.cache_dir / f"riphah_{cache_type}_{datetime.now().strftime('%Y%m%d')}"

    def _load_cached_vectorstore(self):
        """Load previously cached vector store from disk"""
        cache_path = self._get_cache_path("vectorstore")
        if cache_path.exists():
            try:
                print("Loading cached vector store...")
                self.vectorstore = FAISS.load_local(
                    str(cache_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception as e:
                print(f"Cache load failed: {str(e)[:60]}")
        return False

    def _save_vectorstore(self):
        """Save vector store to disk for future use"""
        try:
            cache_path = self._get_cache_path("vectorstore")
            self.vectorstore.save_local(str(cache_path))
            print(f"Vector store cached at {cache_path}")
        except Exception as e:
            print(f"Caching failed: {str(e)[:60]}")

    def load_specific_urls(self, urls=None):
        """Scrape web pages and load documents from URLs"""
        if urls is None:
            urls = self.riphah_urls
        
        all_documents = []
        successful = 0
        failed = 0
        
        print(f"\nLoading Riphah University Data")
        print(f"Total URLs to load: {len(urls)}\n")
        
        # Iterate through each URL and scrape content
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}")
            try:
                loader = WebBaseLoader(url)
                documents = loader.load()
                
                # Only accept documents with meaningful content
                if documents and len(documents[0].page_content) > 100:
                    all_documents.extend(documents)
                    content_len = len(documents[0].page_content)
                    successful += 1
                    print(f"  Loaded {content_len} characters")
                else:
                    failed += 1
                    print(f"  Minimal content (skipped)")
                time.sleep(0.3)
            except Exception as e:
                failed += 1
                print(f"  Error: {str(e)[:50]}")
        
        print(f"\nLoading Summary:")
        print(f"Successful: {successful} | Failed: {failed}")
        print(f"Total documents: {len(all_documents)}\n")
        
        return all_documents

    def process_documents(self, documents):
        """Convert documents into vector embeddings and create searchable index"""
        if not documents:
            print("No documents to process")
            return False
        
        print("Processing documents...")
        
        # Split documents into manageable chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"  Created {len(chunks)} text chunks")
        
        # Create vector embeddings for all chunks
        print("  Creating embeddings...")
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        print("  Vector store created")
        
        self._save_vectorstore()
        self._setup_retriever_and_prompt()
        
        print(f"\nChatbot Ready! Processed {len(chunks)} chunks\n")
        return True

    def _setup_retriever_and_prompt(self):
        """Initialize retriever and prompt template for question answering"""
        # Retriever fetches relevant documents based on query
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        # Prompt template structures how context and question are presented to LLM
        template = f"""{self.SYSTEM_PROMPT}

Context: {{context}}

Question: {{question}}

Answer:"""
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def initialize(self, custom_urls=None, use_cache=True):
        """Initialize chatbot with cached data or fresh scrape"""
        if use_cache and self._load_cached_vectorstore():
            self._setup_retriever_and_prompt()
            return True
        
        # Load and process new data if cache not available
        documents = self.load_specific_urls(custom_urls)
        return self.process_documents(documents)

    def refresh_data(self):
        """Refresh all data by scraping website again"""
        print("\nRefreshing Riphah data...")
        return self.initialize(use_cache=False)

    def chat(self, question):
        """Answer user question using RAG approach"""
        if self.retriever is None or self.prompt is None:
            return "Please initialize the chatbot first"
        
        # Enhance user question with context keywords
        enhanced_question = self._enhance_question(question)
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(enhanced_question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt with context and question
            formatted_prompt = self.prompt.format(
                context=context,
                question=enhanced_question
            )
            
            # Get response from LLM
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)[:100]}"

    def chat_with_sources(self, question):
        """Answer question and return source URLs"""
        if self.retriever is None or self.prompt is None:
            return "Please initialize the chatbot first", []
        try:
            enhanced_question = self._enhance_question(question)
            
            # Retrieve relevant documents
            docs = self.retriever.invoke(enhanced_question)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format and get response
            formatted_prompt = self.prompt.format(
                context=context,
                question=enhanced_question
            )
            
            response = self.llm.invoke(formatted_prompt)
            
            # Extract source URLs from documents
            sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
            
            return response.content, list(set(sources))
        except Exception as e:
            return f"Error: {str(e)[:100]}", []

    def _enhance_question(self, question):
        """Add context keywords to question for better retrieval"""
        keywords = {
            "admission": "admission requirements documents process",
            "mbbs": "MBBS medical program admission campus",
            "campus": "campus facilities location infrastructure",
            "program": "academic program curriculum degree",
            "hostel": "hostel accommodation student housing",
            "fee": "tuition fees cost charges",
            "scholarship": "scholarship financial aid merit",
            "contact": "contact phone email address department"
        }
        
        enhanced = question
        for keyword, context in keywords.items():
            if keyword.lower() in question.lower():
                enhanced = f"{question} [Information about: {context}]"
                break
        
        return enhanced

    def clear_memory(self):
        """Clear conversation history"""
        print("Conversation memory cleared\n")

    def save_conversation(self, filename="conversation_history.json"):
        """Save conversation to file"""
        try:
            with open(filename, 'w') as f:
                json.dump({"note": "Conversation saved"}, f, indent=2)
            print(f"Conversation saved to {filename}\n")
        except Exception as e:
            print(f"Save failed: {str(e)[:60]}\n")


def main():
    """Main chatbot interface"""
    print("\n" + "="*70)
    print("RIPHAH INTERNATIONAL UNIVERSITY CHATBOT")
    print("="*70 + "\n")
    
    # Load API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file!")
        print("Please create a .env file with: OPENAI_API_KEY=sk-proj-your-api-key-here")
        return
    
    print("API Key loaded from .env file")
    print("\nInitializing Riphah chatbot...")
    chatbot = RiphahChatbot(api_key)
    
    if not chatbot.initialize():
        print("Failed to initialize chatbot")
        return
    
    print("="*70)
    print("CHAT WITH RIPHAH BOT")
    print("="*70)
    print("Commands: 'sources' | 'refresh' | 'clear' | 'save' | 'quit'")
    print("="*70 + "\n")
    
    # Interactive chat loop
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        
        # Handle special commands
        if user_input.lower() == 'quit':
            print("\nThank you for using Riphah Chatbot!")
            break
        elif user_input.lower() == 'refresh':
            chatbot.refresh_data()
            continue
        elif user_input.lower() == 'clear':
            chatbot.clear_memory()
            continue
        elif user_input.lower() == 'save':
            chatbot.save_conversation()
            continue
        elif user_input.lower() == 'sources':
            question = input("Ask your question: ").strip()
            if question:
                print("\nThinking...\n")
                answer, sources = chatbot.chat_with_sources(question)
                print(f"Bot: {answer}\n")
                if sources:
                    print("Sources:")
                    for source in sources:
                        print(f"  - {source}")
                print("\n" + "-"*70 + "\n")
            continue
        
        # Standard chat response
        print("\nThinking...\n")
        response = chatbot.chat(user_input)
        print(f"Bot: {response}\n")
        print("-"*70 + "\n")


if __name__ == "__main__":
    main()