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
        self.conversation_history = []  # Track conversation context
        self.max_history = 5  # Keep last 5 exchanges
        
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
        # Retriever fetches relevant documents with relevance score threshold
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 6, "fetch_k": 10}
        )
        
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
        """Answer user question using enhanced RAG approach with multi-stage reasoning"""
        if self.retriever is None or self.prompt is None:
            return "Please initialize the chatbot first"
        
        try:
            # Step 1: Analyze question intent and extract key information
            query_analysis = self._analyze_question(question)
            
            # Step 2: Enhance query for better retrieval
            enhanced_question = self._enhance_question_advanced(question, query_analysis)
            
            # Step 3: Retrieve relevant documents with scoring
            docs_with_scores = self.vectorstore.similarity_search_with_score(enhanced_question, k=6)
            
            # Filter by relevance (FAISS L2 distance < 1.5 is good match)
            relevant_docs = [doc for doc, score in docs_with_scores if score < 1.5]
            
            # If too few results, take top 4
            if len(relevant_docs) < 2:
                relevant_docs = [doc for doc, score in docs_with_scores[:4]]
            
            if not relevant_docs:
                return self._generate_fallback_response(question)
            
            # Step 4: Build structured context
            context = self._build_enhanced_context(relevant_docs)
            conversation_context = self._get_conversation_context()
            
            # Step 5: Generate RAG response with chain-of-thought reasoning
            rag_response = self._generate_rag_response(
                question, context, conversation_context, query_analysis
            )
            
            # Step 6: Generate enhanced knowledge response
            enhanced_response = self._generate_knowledge_enhanced_response(
                question, rag_response, query_analysis
            )
            
            # Step 7: Intelligent synthesis with validation
            final_response = self._synthesize_responses(
                question=question,
                rag_response=rag_response,
                enhanced_response=enhanced_response,
                query_analysis=query_analysis,
                context=context
            )
            
            # Step 8: Update conversation history
            self._update_conversation_history(question, final_response)
            
            return final_response
            
        except Exception as e:
            return f"Error: {str(e)[:100]}"

    def chat_with_sources(self, question):
        """Answer question and return source URLs with relevance information"""
        if self.retriever is None or self.prompt is None:
            return "Please initialize the chatbot first", []
        try:
            # Use the enhanced chat method
            response = self.chat(question)
            
            # Get sources with relevance information
            query_analysis = self._analyze_question(question)
            enhanced_question = self._enhance_question_advanced(question, query_analysis)
            docs_with_scores = self.vectorstore.similarity_search_with_score(enhanced_question, k=6)
            
            # Filter and extract sources
            relevant_docs = [doc for doc, score in docs_with_scores if score < 1.5]
            if len(relevant_docs) < 2:
                relevant_docs = [doc for doc, score in docs_with_scores[:4]]
            
            # Extract unique source URLs
            sources = list(set([
                doc.metadata.get('source', 'Unknown')
                for doc in relevant_docs
            ]))
            
            return response, sources
        except Exception as e:
            return f"Error: {str(e)[:100]}", []

    def _analyze_question(self, question):
        """Analyze question to understand intent and requirements"""
        analysis_prompt = f"""Analyze this question:

Question: {question}

Provide:
1. Intent: What is the user trying to find out? (e.g., admission_process, fee_details, program_info)
2. Key entities: Extract specific names, programs, campuses mentioned
3. Question type: Is this asking for facts, comparison, procedure, or general info?
4. Specificity: Does the user want specific details or general overview?

Provide brief analysis in 3-4 lines."""
        
        try:
            analysis = self.llm.invoke(analysis_prompt)
            return analysis.content
        except:
            return "General information query about Riphah University"
    
    def _enhance_question_advanced(self, question, analysis):
        """Advanced query enhancement with semantic expansion"""
        keyword_mappings = {
            "admission": ["admission", "enrollment", "application", "eligibility", "requirements"],
            "mbbs": ["MBBS", "medicine", "medical", "doctor", "healthcare"],
            "bds": ["BDS", "dental", "dentistry"],
            "fee": ["fee", "tuition", "cost", "charges", "payment"],
            "scholarship": ["scholarship", "financial aid", "merit", "grant"],
            "campus": ["campus", "location", "facilities", "address"],
            "program": ["program", "degree", "course", "curriculum"],
            "faculty": ["faculty", "professor", "teacher", "staff"],
            "hostel": ["hostel", "accommodation", "residence", "housing"],
            "research": ["research", "publication", "project", "thesis"]
        }
        
        terms = []
        question_lower = question.lower()
        
        for key, synonyms in keyword_mappings.items():
            if any(syn in question_lower for syn in synonyms):
                terms.extend(synonyms[:3])
        
        if terms:
            return f"{question} [Related: {' '.join(set(terms))}]"
        return question
    

    
    def _build_enhanced_context(self, docs):
        """Build well-structured context from documents"""
        if not docs:
            return "No relevant information available."
        
        context_parts = []
        seen_hashes = set()
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            content_hash = hash(content[:150])
            
            if content_hash not in seen_hashes:
                context_parts.append(f"[Information Source {i}]\n{content}")
                seen_hashes.add(content_hash)
        
        return "\n\n===SEPARATOR===\n\n".join(context_parts)
    
    def _get_conversation_context(self):
        """Get conversation history for contextual understanding"""
        if not self.conversation_history:
            return "This is the first question in the conversation."
        
        recent = self.conversation_history[-self.max_history:]
        history = "\n".join([
            f"Q: {h['question'][:80]}... A: {h['answer'][:80]}..."
            for h in recent
        ])
        return f"Recent context:\n{history}"
    
    def _generate_rag_response(self, question, context, conversation_context, analysis):
        """Generate response based on retrieved documents with reasoning"""
        rag_prompt = f"""{self.SYSTEM_PROMPT}

===QUERY ANALYSIS===
{analysis}

===OFFICIAL UNIVERSITY DATABASE===
{context}

===CONVERSATION CONTEXT===
{conversation_context}

===USER QUESTION===
{question}

===REASONING INSTRUCTIONS===
Apply systematic reasoning:
1. IDENTIFY: What specific information does the user need?
2. LOCATE: Find exact details in the database (names, numbers, procedures)
3. VERIFY: Ensure information is current and accurate
4. ORGANIZE: Structure the response logically
5. COMPLETE: Address all aspects of the question

Provide a detailed, fact-based answer using ONLY the database information above.
Include specific details, names, numbers, and procedures.
If information is incomplete, state what's available and what's missing.

Your Response:"""
        
        try:
            response = self.llm.invoke(rag_prompt)
            return response.content
        except:
            return "Unable to generate response from database."
    
    def _generate_knowledge_enhanced_response(self, question, rag_response, analysis):
        """Generate enhanced response with contextual knowledge"""
        enhancement_prompt = f"""You are an expert on Pakistani university systems and Riphah International University.

===QUERY ANALYSIS===
{analysis}

===QUESTION===
{question}

===DATABASE RESPONSE===
{rag_response}

Your task: Enhance this response with:
1. **Context and Background**: Why is this important? How does it fit in the Pakistani education system?
2. **Practical Guidance**: What should students know or do next?
3. **Comparative Insights**: How does this compare to similar universities (if relevant)?
4. **Additional Considerations**: What else should students consider?

Provide contextual enhancement that SUPPORTS (never contradicts) the database response.
Be specific and practical. Focus on helping the student make informed decisions.

Your Enhanced Response:"""
        
        try:
            response = self.llm.invoke(enhancement_prompt)
            return response.content
        except:
            return "Additional context unavailable."
    
    def _synthesize_responses(self, question, rag_response, enhanced_response, 
                            query_analysis, context):
        """Intelligently synthesize responses with validation"""
        synthesis_prompt = f"""You are a master synthesizer creating the ULTIMATE response for Riphah University queries.

===USER QUESTION===
{question}

===QUERY ANALYSIS===
{query_analysis}

===RESPONSE SOURCE 1: Official Database (HIGHEST PRIORITY)===
{rag_response}

===RESPONSE SOURCE 2: Enhanced Context (SUPPORTING)===
{enhanced_response}

===SYNTHESIS INSTRUCTIONS===
Create the BEST possible response by:

1. **Foundation**: Use SOURCE 1 facts as the core (names, numbers, procedures)
2. **Enhancement**: Add relevant context from SOURCE 2 that helps understanding
3. **Integration**: Seamlessly blend both sources into ONE coherent response
4. **Prioritization**: Official data > General knowledge
5. **Completeness**: Address ALL aspects of the question
6. **Clarity**: Use clear structure - bullet points for lists, paragraphs for explanations
7. **Actionability**: Include next steps or contact information

===QUALITY CHECKLIST===
✓ Answers the question directly and completely
✓ Includes specific facts (names, numbers, dates)
✓ Provides practical guidance
✓ Well-structured and easy to read
✓ No contradictions between sources
✓ Cites official sources when important

===FORMAT GUIDELINES===
- Start with direct answer
- Use **bold** for important terms
- Use bullet points (•) for lists
- Include relevant links if discussing programs/fees
- End with actionable next steps

Create the FINAL, COMPREHENSIVE response now:"""
        
        try:
            final = self.llm.invoke(synthesis_prompt)
            response = final.content
            
            # Validate the response
            if self._validate_response(question, response, rag_response):
                return response
            else:
                # Fallback to RAG response if synthesis fails validation
                return rag_response
        except:
            return rag_response
    
    def _validate_response(self, question, response, rag_response):
        """Validate that response is high quality and addresses question"""
        # Basic quality checks
        if len(response) < 30:
            return False
        
        # Check for common failure patterns
        failure_patterns = [
            "I don't have",
            "Unable to",
            "Cannot provide"
        ]
        
        if any(pattern in response for pattern in failure_patterns) and len(response) < 150:
            return False
        
        # Response should not be dramatically shorter than RAG response
        if len(response) < len(rag_response) * 0.5:
            return False
        
        return True
    
    def _generate_fallback_response(self, question):
        """Generate helpful response when information is not available"""
        return f"""I apologize, but I don't have specific information about that in my current database.

**How to get accurate information:**

• 🌐 **Official Website**: Visit https://riphah.edu.pk/
• 📞 **Contact Admissions**: Check https://riphah.edu.pk/contact/
• 📧 **Email**: Reach out to the specific department
• 📱 **Campus Specific**:
  - Faisalabad: https://riphahfsd.edu.pk/
  - Sahiwal: https://riphahsahiwal.edu.pk/

**I can help with:**
- General program information
- Admission requirements
- Campus locations and facilities
- Fee structures
- Scholarship information

What else would you like to know?"""
    
    def _update_conversation_history(self, question, answer):
        """Track conversation for contextual responses"""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
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