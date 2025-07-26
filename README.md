# Multilingual Bangla-English RAG System using OCR, FAISS & Transformers

This project implements a *Retrieval-Augmented Generation (RAG)* pipeline in *Google Colab* that supports both *Bengali and English* input. The system extracts scanned PDF text via OCR, creates semantic embeddings using multilingual models, retrieves context using FAISS, and generates answers using a Transformer QA model. An optional *FastAPI endpoint* is also provided.



## Setup Guide (Google Colab)

### ✅ Step 1: Install OCR and Language Data
!apt update && apt install -y tesseract-ocr tesseract-ocr-ben
!wget -O ben.traineddata https://github.com/tesseract-ocr/tessdata/raw/main/ben.traineddata
!mkdir -p /content/tessdata
!mv ben.traineddata /content/tessdata/
`

### ✅ Step 2: Install Required Python Packages

!pip install pytesseract pillow pymupdf
!pip install sentence-transformers faiss-cpu transformers fastapi uvicorn

### ✅ Step 3: Upload Text File

* Upload your Bangla_OCR_Full_Text.txt containing the OCR-extracted text to the working directory.

### ✅ Step 4: Run the Main Code

* Use the provided script in main.py or notebook cells.

---

## 📦 Used Tools, Libraries & Packages

| Package/Tool            | Purpose                                 |
| ----------------------- | --------------------------------------- |
| pytesseract           | OCR extraction from scanned images/PDF  |
| tesseract-ocr-ben     | Bengali OCR language support            |
| pillow, pymupdf     | Image and PDF handling                  |
| sentence-transformers | Text embedding (multilingual)           |
| faiss-cpu             | Vector similarity search                |
| transformers          | FLAN-T5 question-answering              |
| fastapi, uvicorn    | REST API deployment (optional)          |
| scikit-learn          | Cosine similarity evaluation (optional) |

---

## 💬 Sample Queries and Outputs

*Input Question:*

অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?

*Generated Answer:*

শুম্ভুনাথ
---
---

## 📡 API Documentation (Optional)

Running the FastAPI version:

### ✅ Endpoint

POST /ask

### ✅ Request Example

{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}

### ✅ Response Example

{
  "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শুম্ভুনাথ",
  "context": "...(retrieved context)...",
  "grounded": true
}

### ✅ Local Run Command

uvicorn main:app --reload

---

## 📊 Evaluation Matrix (Optional)

We use *cosine similarity* to evaluate semantic closeness between the user query and the retrieved document chunks.

### ✅ Sample Code

from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_score(query, chunks):
    q_vec = embed_model.encode([query])
    c_vecs = embed_model.encode(chunks)
    return cosine_similarity(q_vec, c_vecs)[0]

* Values close to 1 indicate higher semantic alignment.

---

## Answers to the given questions

### 1️⃣ *What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?*

We used pytesseract with the *Tesseract OCR engine* and the **Bengali language model (ben.traineddata)**. Tesseract supports Bengali well and can extract text from scanned documents.

*Challenges:*

* OCR often introduces formatting errors such as broken lines, irregular spacing, and incorrect ligatures.
* Text post-processing was required to normalize formatting and improve consistency for chunking and embedding.

---

### 2️⃣ *What chunking strategy did you choose? Why do you think it works well for semantic retrieval?*

We used a *character-based chunking* approach with the following parameters:

* *Chunk size*: 500 characters
* *Overlap*: 100 characters

*Why it works:*

* Ensures each chunk is small enough for transformer models without losing meaning.
* Overlapping helps preserve context across adjacent chunks.
* It's effective when paragraph or sentence boundaries are distorted during OCR.

---

### 3️⃣ *What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?*

We used:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

*Why:*

* It supports 100+ languages, including Bengali and English.
* Lightweight, fast, and works well on limited resources like Colab.
* Trained to capture *semantic similarity* between multilingual sentences.

*How it works:*

* Transforms text into dense vector embeddings using attention mechanisms.
* Captures semantic meaning by placing similar sentences near each other in vector space.

---

### 4️⃣ *How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?*

We:

* Encode both the *user query* and *document chunks* into vectors.
* Use *FAISS* with *L2 similarity* to retrieve the top-k most relevant chunks.

*Why FAISS:*

* Optimized for fast and scalable similarity search.
* Efficient even with thousands of embeddings.
* Simple to use and well-integrated with sentence-transformers.

---

### 5️⃣ *How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?*

*How do we ensure meaningful comparison:*

* Use the same embedding model for both questions and document chunks.
* This ensures both exist in the same semantic space.
* RAG retrieves the most similar chunks and passes them with the question to the *FLAN-T5* model.

*If the query is vague:*

* The system may return irrelevant or partial results.
* Can be improved by:

  * Using query rephrasing or clarification.
  * Retrieving more chunks.
  * Adding user history or context enrichment.

---

### 6️⃣ *Do the results seem relevant? If not, what might improve them?*

Yes, the system generates *highly relevant answers* for well-formed questions.

*Ways to improve further:*

* *Better chunking*: Use sentence or paragraph-based chunking if formatting allows.
* *Stronger embeddings*: Use e5-mistral, LaBSE, or instruction-tuned models.
* *Larger QA model*: Upgrade from flan-t5-base to flan-t5-large for richer answers.
* *Layout-aware OCR*: Use LayoutLM or Donut for accurate document parsing.

---

