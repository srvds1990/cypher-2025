# Cypher Sessions – Agentic AI App

This project is a Streamlit-based Agentic AI app that:
- Ingests session transcripts
- Creates embeddings in ChromaDB
- Summarizes sessions using GPT
- Answers user questions with context
- Visualizes keywords interactively

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/YOUR-USERNAME/cypher-sessions.git
cd cypher-sessions

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run ui/app.py
