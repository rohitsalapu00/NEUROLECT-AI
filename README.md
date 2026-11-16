# NEUROLECT AI
## Intelligent Lecture Understanding, Question Generation & Automated Assessment System

NEUROLECT AI is a genuine transformer-based educational platform that ingests lecture recordings and study documents to produce structured summaries, topic taxonomies, named entity graphs, educational concept breakdowns, multi-type assessment questions, and an interactive quiz assessment engine with semantic evaluation and learning analytics.

---

### Key Architectural Tenet: 100% RAG-Free

**NEUROLECT AI is completely RAG-free.**
- No vector databases (FAISS, ChromaDB, Pinecone, Weaviate).
- No embedding-based document retrieval or similarity document search.
- No LangChain or LlamaIndex RAG workflows.
- Instead, content is processed through direct text extraction, sentence segmentation, token-aware chunking, and direct pretrained transformer inference.
- Sentence Transformers are utilized **strictly for semantic evaluation of student answers and duplicate question detection**, never for document retrieval.

---

### Core Capabilities

1. **Equal Priority Multi-Modal Ingestion**:
   - **Audio Lectures**: MP3, WAV, M4A via OpenAI Whisper speech recognition with timestamp tracking.
   - **Video Lectures**: MP4, MOV, MKV via FFmpeg audio extraction followed by Whisper transcription.
   - **PDF & Plain Text**: PDF documents via PyMuPDF with page preservation, plus direct text uploads and pasting.
2. **Unified Content Representation**:
   - All modalities are mapped into a standardized `LectureContent` object with chunked text and provenance metadata.
3. **Deep Lecture Understanding**:
   - **Transformer Summarization**: DistilBART / BART large models producing concise, detailed, and topic-wise lecture notes.
   - **Topic & Subtopic Detection**: Transparent hierarchical topic structuring.
   - **Keyword Extraction**: KeyBERT with maximal marginal relevance.
   - **Named Entity Recognition**: Pretrained BERT-NER (`dslim/bert-base-NER`) identifying people, organizations, locations, and technical entities.
   - **Educational Concept Extraction**: Grounded definitions, explanations, and core principles mapped to source timestamps/pages.
4. **AI-Powered Question Generation**:
   - Instruction-tuned Flan-T5 model generating 6 question categories:
     - Multiple Choice Questions (MCQ) with 3 plausible distractors.
     - Short Answer.
     - Descriptive.
     - Conceptual.
     - Application-based.
     - Analytical.
   - **Bloom's Taxonomy Classification**: 6 cognitive levels (Remember, Understand, Apply, Analyze, Evaluate, Create).
   - **Difficulty Grading**: Easy, Medium, Hard.
   - **Question Quality Validation**: Deduplication, format checks, and grounding validation.
5. **Interactive Assessment & Quiz Engine**:
   - Timed or self-paced quizzes.
   - Immediate feedback and explanations.
6. **Semantic Answer Evaluation**:
   - `sentence-transformers/all-MiniLM-L6-v2` compares student answers against reference solutions evaluating semantic similarity, concept coverage, and keyword alignment.
7. **Learning Analytics & History**:
   - SQLite persistence layer.
   - Topic mastery profiling, Bloom's cognitive distribution, weak concept diagnostics, and targeted revision recommendations.

---

### Technology Stack

| Layer | Technologies |
|---|---|
| **Programming** | Python 3.13 |
| **User Interface** | Streamlit, Plotly, HTML/CSS |
| **Speech & Audio** | OpenAI Whisper, FFmpeg / ImageIO-FFmpeg |
| **Document Parsing** | PyMuPDF (fitz) |
| **Deep Learning** | PyTorch (CPU-optimized), Hugging Face Transformers |
| **NLP & Semantics** | NLTK, KeyBERT, Sentence-Transformers, scikit-learn |
| **Database** | SQLite3 |

---

### Installation & Quick Start

```powershell
# 1. Clone repository & navigate to directory
cd D:\projects\NEUROLECT_AI

# 2. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 3. Launch Streamlit Application
streamlit run app.py
```
