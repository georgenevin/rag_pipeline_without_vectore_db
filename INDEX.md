# 🚀 Infopark RAG System - Getting Started

Welcome! You now have a complete Retrieval-Augmented Generation (RAG) system for querying Infopark news using embeddings and Mistral AI.

## 📚 Documentation Index

Start here based on your need:

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **QUICK_REFERENCE.md** | Quick commands and examples | 5 min |
| **README_RAG.md** | Complete feature documentation | 15 min |
| **WINDOWS_SETUP.md** | Step-by-step Windows setup | 10 min |
| **RAG_USAGE.md** | Detailed usage guide and API | 15 min |

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Set Mistral API Key
```powershell
$env:MISTRAL_API_KEY = "your-mistral-api-key"
```

### 3. Run the System
```powershell
python main.py
```

That's it! You now have an interactive RAG system.

## 🎯 What to Do Next

### For Beginners
1. Read **QUICK_REFERENCE.md** (5 min)
2. Run `python main.py` and ask a question
3. Try `python query.py --examples` to see more

### For Windows Users
1. Follow **WINDOWS_SETUP.md** for detailed steps
2. Create batch files for quick access
3. Set up VSCode integration

### For Developers
1. Read **README_RAG.md** for full features
2. Check **rag_examples.py** for code patterns
3. Customize `rag.py` for your needs

## 📁 Project Files

### Core Application
- **rag.py** - Main RAG system class
- **main.py** - Interactive entry point
- **query.py** - Command-line utility
- **rag_examples.py** - Advanced examples

### Data
- **embeddings.jsonl** - Pre-computed embeddings from news articles
- **infopark_news.md** - Source news content

### Configuration
- **requirements.txt** - Python dependencies
- **pyproject.toml** - Project metadata

### Documentation
- **README_RAG.md** - Full documentation
- **RAG_USAGE.md** - Detailed usage guide
- **QUICK_REFERENCE.md** - Cheat sheet
- **WINDOWS_SETUP.md** - Windows setup guide
- **INDEX.md** - This file

## 🚀 Usage Modes

### Mode 1: Interactive Chat
```bash
python main.py
```
Ask questions naturally and get contextual answers.

### Mode 2: Single Query
```bash
python query.py "What is Infopark Phase 3?"
```
Get quick answers from command line.

### Mode 3: Advanced Examples
```bash
python rag_examples.py
```
Choose from predefined examples and batch processing.

### Mode 4: Python Integration
```python
from rag import EmbeddingRAG
rag = EmbeddingRAG()
response = rag.query("Your question here")
```

## ❓ Common Questions

### Q: Do I need to generate embeddings?
**A:** No! Embeddings are already pre-computed in `embeddings.jsonl`. Just use the system as-is.

### Q: Where do I get the Mistral API key?
**A:** Sign up at [console.mistral.ai](https://console.mistral.ai/) and create an API key.

### Q: How many tokens/requests can I make?
**A:** Depends on your Mistral plan. Check your console for limits.

### Q: Can I add more news articles?
**A:** Yes! Generate embeddings for new articles and append to `embeddings.jsonl` in the same JSON format.

### Q: What if I'm on Windows?
**A:** Follow **WINDOWS_SETUP.md** for step-by-step instructions.

## 🔧 Troubleshooting

### Error: "MISTRAL_API_KEY not set"
Set your API key before running:
```powershell
$env:MISTRAL_API_KEY = "your-key"
```

### Error: "Embeddings file not found"
Make sure `embeddings.jsonl` exists in the project root.

### Error: "ModuleNotFoundError"
Reinstall dependencies:
```bash
pip install -r requirements.txt
```

For more issues, see **WINDOWS_SETUP.md** or **README_RAG.md**.

## 📊 How RAG Works

```
Your Question
     ↓
Similarity Search (find relevant docs)
     ↓
Retrieved Context
     ↓
LLM Prompt (question + context)
     ↓
Mistral AI
     ↓
Answer
```

The system finds the most relevant news articles and uses them as context for the LLM to generate accurate answers.

## 💡 Example Queries to Try

- "What did the CM announce about Lulu Group?"
- "Tell me about Infopark Phase 3"
- "How many jobs will be created?"
- "What is the i by Infopark?"
- "Summarize recent developments"
- "What investments were announced?"
- "How is Kochi ranked globally?"
- "Tell me about GCC Surge 2025"

## 🎓 Learning Path

```
Start Here
    ↓
QUICK_REFERENCE.md (5 min)
    ↓
Run: python main.py (try it!)
    ↓
Try: python query.py --examples
    ↓
Read: README_RAG.md (optional - more details)
    ↓
Customize: Modify rag.py for your needs
```

## ⚙️ Configuration

### Quick Configuration Changes

```python
# In your script or rag.py:
rag = EmbeddingRAG()

# Retrieve more documents for better context
response = rag.query(query, top_k=5)

# Adjust LLM creativity (0=consistent, 1=creative)
rag.llm.temperature = 0.3

# Limit response length
rag.llm.max_tokens = 500
```

## 🔗 Useful Links

- [Mistral AI Console](https://console.mistral.ai/)
- [Mistral API Docs](https://docs.mistral.ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAG Overview](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

## 📞 Need Help?

1. **Quick answers?** → Check **QUICK_REFERENCE.md**
2. **Setup issues?** → Read **WINDOWS_SETUP.md**
3. **How does RAG work?** → See **README_RAG.md**
4. **Code examples?** → Run `python rag_examples.py`
5. **Detailed guide?** → Read **RAG_USAGE.md**

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Mistral API key obtained
- [ ] Environment variable set: `$env:MISTRAL_API_KEY`
- [ ] Embeddings file exists: `embeddings.jsonl`
- [ ] Can import: `python -c "from rag import EmbeddingRAG"`
- [ ] Can run: `python main.py`

## 🚀 Ready to Start?

### Option 1: Interactive Mode (Recommended for beginners)
```bash
python main.py
```

### Option 2: Quick Query
```bash
python query.py "What is Infopark Phase 3?"
```

### Option 3: See Examples
```bash
python query.py --examples
```

Pick one and start exploring!

---

## 📝 File Descriptions

### Application Files
- **rag.py** (370 lines) - Core RAG system with embedding loading, similarity search, and LLM integration
- **main.py** (20 lines) - Simple interactive entry point
- **query.py** (120 lines) - Command-line interface for single queries
- **rag_examples.py** (200 lines) - Advanced usage patterns and batch processing

### Data Files
- **embeddings.jsonl** - 10+ pre-computed embeddings from news articles
- **infopark_news.md** - Source news content used to generate embeddings

### Documentation (1000+ lines total)
- **README_RAG.md** - Comprehensive feature guide
- **RAG_USAGE.md** - Detailed setup and usage
- **QUICK_REFERENCE.md** - Quick commands and tips
- **WINDOWS_SETUP.md** - Windows-specific setup
- **INDEX.md** - This file

## 🎉 That's It!

You're all set! Choose a usage mode above and start querying your Infopark news database.

Happy exploring! 🚀

---

**System Status:** ✅ Ready to Use  
**Version:** 1.0  
**Last Updated:** 2025  
**Documentation:** Complete
