# 🎓 Shanyan AI — Viva Exam Q&A Guide

## Prepared for: Shankar Narayanan, Student, Dr. MGR University

---

## Section 1: Project Overview

### Q1. What is the name and purpose of your project?

**A:** The project is called **Shanyan AI – Auto Ticket Classification and Reply System**. It automates customer support by:

1. Accepting a customer's complaint or query in plain English
2. Classifying it into the correct support queue (e.g., Technical Support, Billing, Sales) using a fine-tuned DistilBERT model
3. Automatically generating a professional acknowledgement reply using Google's Gemma 3 27B model (via the Gemini API)
4. Routing the ticket to the appropriate department

### Q2. What real-world problem does this project solve?

**A:** In companies, customer support tickets are received in large volumes. Manually reading, categorizing, assigning to the right department, and drafting a first reply takes a lot of time and human effort. My project automates all three steps:

- **Classification** → Instant routing to the correct team
- **Auto-reply** → Customer gets an immediate acknowledgment
- **Department assignment** → No manual triage needed

This reduces response time from hours/days to seconds.

### Q3. What is the overall architecture of the system?

**A:** It follows a **3-tier architecture**:

1. **Frontend (React + Vite)** → User interface where clients submit tickets, and staff view their department's tickets
2. **Backend (FastAPI + Python)** → REST API that handles authentication, ticket creation, ML prediction, and auto-reply generation
3. **Database (SQLite via SQLAlchemy)** → Stores users and tickets

Additionally, the backend has two AI services:

- **TicketClassifierService** → Uses a fine-tuned DistilBERT model for classification
- **GeminiService** → Uses Google Gemma 3 27B for generating replies

### Q4. What is the full flow when a customer submits a ticket?

**A:** The flow follows 5 clear steps:

1. **User submits** a description through the React frontend
2. **Frontend sends** a POST request to `/api/predict`
3. **Backend classifies** the text using the fine-tuned DistilBERT model → gets a queue like "Technical Support"
4. **Backend maps** the queue to a department (e.g., Technical Support → `technical_support`)
5. **Backend generates** an auto-reply using Google Gemma 3 27B (via Gemini API) with a prompt that includes the ticket details
6. **Backend saves** the ticket to the SQLite database with all details
7. **Frontend displays** the AI-generated reply and ticket number to the user

---

## Section 2: Machine Learning — Model Selection

### Q5. What ML model did you use for ticket classification and why?

**A:** I used **DistilBERT** (`distilbert-base-uncased`), which is a lighter version of BERT.

**Why DistilBERT specifically:**
| Feature | BERT-base | DistilBERT |
|---------|-----------|------------|
| Parameters | 110M | 66M (40% smaller) |
| Speed | 1× | 1.6× faster |
| Accuracy | Baseline | Retains 97% of BERT's accuracy |
| Size on disk | ~420 MB | ~260 MB |

- It is **pre-trained** on a large English corpus (Wikipedia + Book Corpus), so it already understands language
- It is **small enough** to fine-tune on a regular machine without a GPU
- It is **accurate enough** for text classification tasks like ours
- It supports **transfer learning** — we only need to train the final classification layer

### Q6. What is DistilBERT? How does it differ from BERT?

**A:** DistilBERT is created using a technique called **Knowledge Distillation**:

- A large "teacher" model (BERT) teaches a smaller "student" model (DistilBERT)
- The student learns to mimic the teacher's output probability distributions
- Result: 40% fewer parameters, 60% faster, while retaining 97% accuracy

Key differences:

- BERT has **12 transformer layers**; DistilBERT has **6 layers**
- DistilBERT removes the **token-type embeddings** and the **pooler layer**
- DistilBERT uses a **triple loss function** during distillation: distillation loss + masked language modeling loss + cosine embedding loss

### Q7. Why not use a larger model like GPT-4 or LLaMA for classification?

**A:** Several reasons:

1. **Overkill** — Classification is a simple task (mapping text → one of 10 categories). A 66M parameter model is sufficient.
2. **Speed** — DistilBERT classifies a ticket in milliseconds; LLMs take seconds
3. **Cost** — No API call needed for classification; the model runs locally for free
4. **Reliability** — A fine-tuned classification model gives consistent, deterministic results, unlike LLMs which can vary
5. **Offline capable** — Works without internet since the model is saved locally

### Q8. What is Transfer Learning? How is it used here?

**A:** Transfer learning means taking a model that was **pre-trained on a large general dataset** and **fine-tuning it on your specific, smaller dataset**.

In our project:

- **Pre-trained part**: DistilBERT was trained on Wikipedia + BookCorpus (3.3 billion words) to understand English language structure
- **Fine-tuned part**: We take this pre-trained model and train only the **classification head** (the final layers) on our customer support ticket dataset

Benefits:

- We don't need millions of training samples — even 500-5000 samples give good results
- Training is fast (minutes, not days)
- The model already "understands" English; we just teach it to recognize ticket categories

### Q9. What model is used for auto-reply generation and why?

**A:** We use **Google Gemma 3 27B-IT** (Instruction-Tuned) via the Gemini API.

Why this model:

- **27 billion parameters** — powerful enough for natural, professional text generation
- **Instruction-tuned** — specifically trained to follow instructions and generate formatted responses
- **Free tier available** — Gemini API provides free access for development
- **High quality** — Generates professional, contextual customer support replies
- **It is different from classification** — While DistilBERT classifies, we need a **generative model** to write replies, which is exactly what LLMs excel at

### Q10. Why use two different models instead of one?

**A:** This is a key design decision called **using the right tool for the job**:

| Task             | Model                    | Why                                                   |
| ---------------- | ------------------------ | ----------------------------------------------------- |
| Classification   | DistilBERT (66M params)  | Fast, accurate, runs locally, deterministic           |
| Reply Generation | Gemma 3 27B (27B params) | Creative text generation needs a large generative LLM |

Using one large LLM for everything would be:

- **Slower** — Each classification would need an API call
- **Expensive** — API calls cost money at scale
- **Less reliable** — LLMs can give inconsistent category labels
- **Dependent on internet** — Can't classify offline

---

## Section 3: Training Process

### Q11. What dataset did you use for training?

**A:** I used the **"Tobi-Bueck/customer-support-tickets"** dataset from Hugging Face. It contains real customer support tickets with their categories.

- **Total samples**: ~43,000+ tickets
- **After filtering**: Kept the **top 10 most common queues/categories**
- **Categories include**: Technical Support, Product Support, Customer Service, IT Support, Billing and Payments, Returns and Exchanges, Sales and Pre-Sales, Human Resources, Service Outages and Maintenance, General Inquiry

### Q12. Explain the training pipeline step by step.

**A:**

```
Step 1: Load Dataset     → Load from Hugging Face Hub
Step 2: Filter           → Keep only top 10 queue categories
Step 3: Label Encoding   → Convert category names to numbers (0-9)
                           using scikit-learn's LabelEncoder
Step 4: Tokenization     → Convert text to token IDs using DistilBERT tokenizer
                           (max_length=128, padding, truncation)
Step 5: Model Setup      → Load pre-trained DistilBERT + add classification head
                           with num_labels=10
Step 6: Training         → Fine-tune using Hugging Face Trainer with:
                           - AdamW optimizer
                           - Weight decay = 0.01
                           - Learning rate warmup
Step 7: Save             → Save model weights, tokenizer, and label encoder
```

### Q13. What is tokenization and why is it important?

**A:** Tokenization is the process of converting human-readable text into **numerical token IDs** that the model can process.

Example:

```
Input:  "My laptop won't start"
Tokens: ["my", "laptop", "won", "'", "t", "start"]
IDs:    [2026, 12191, 2180, 1005, 1056, 2707]
```

We use DistilBERT's **WordPiece tokenizer**, which:

- Breaks words into **subword units** (handles unknown words)
- Adds special tokens: `[CLS]` at the start, `[SEP]` at the end
- **Pads** shorter texts to a fixed length (128 tokens)
- **Truncates** longer texts to 128 tokens

### Q14. What is Label Encoding? Why did you use it?

**A:** Label Encoding converts **categorical text labels into numbers**:

```
"Technical Support"  → 0
"Product Support"    → 1
"Billing"            → 2
... and so on
```

The ML model outputs a number (0-9). The label encoder lets us convert this **back to the category name** during prediction. We save the encoder as `label_encoder.pkl` using Python's `pickle` module.

### Q15. What are the training hyperparameters and why were they chosen?

**A:**

| Hyperparameter   | Local Value | Docker Value | Purpose                                                 |
| ---------------- | ----------- | ------------ | ------------------------------------------------------- |
| Epochs           | 3           | 1            | Number of full passes through data                      |
| Batch Size       | 16          | 8            | Samples processed together (trades memory for speed)    |
| Warmup Steps     | 500         | 10           | Gradually increases learning rate to avoid wild updates |
| Weight Decay     | 0.01        | 0.01         | L2 regularization to prevent overfitting                |
| Max Token Length | 128         | 128          | Most tickets are under 128 tokens; saves memory         |
| Eval Strategy    | per epoch   | none         | Skip evaluation in Docker to speed up                   |

### Q16. What evaluation metrics did you use?

**A:** Four standard classification metrics:

- **Accuracy** — Overall percentage of correct predictions
- **Precision** — Of all tickets predicted as category X, how many were actually X?
- **Recall** — Of all tickets that are actually category X, how many did we correctly identify?
- **F1 Score** — Harmonic mean of precision and recall (balanced metric)

We use `weighted` averaging because the categories have different numbers of samples.

### Q17. What is an Epoch? What is a Batch?

**A:**

- **Epoch** = One complete pass through the entire training dataset. If we have 500 samples, one epoch means the model has seen all 500 samples once.
- **Batch** = A subset of samples processed together in one step. With batch_size=8 and 500 samples: one epoch = 500/8 ≈ 63 steps.
- More epochs → model learns more but risks **overfitting** (memorizing instead of learning)

### Q18. What is Overfitting? How do you prevent it?

**A:** Overfitting is when the model performs very well on training data but poorly on new, unseen data — it has "memorized" the training examples instead of learning general patterns.

Prevention techniques used in this project:

1. **Weight Decay (0.01)** — Adds a penalty for large weights (L2 regularization)
2. **Train/Test Split (90/10)** — Evaluates on data the model hasn't seen during training
3. **Limited Epochs** — Not training for too long (3 epochs locally, 1 in Docker)
4. **Pre-trained model** — Transfer learning naturally regularizes because the model starts with good general knowledge

---

## Section 4: Inference & Classification

### Q19. How does the classification work at runtime (inference)?

**A:**

```python
# Step 1: Tokenize the input text
inputs = tokenizer("My laptop won't start", return_tensors="pt", max_length=128)

# Step 2: Pass through model (no gradient computation needed)
with torch.no_grad():
    outputs = model(**inputs)

# Step 3: Get predicted class (highest probability)
predicted_class_id = torch.argmax(outputs.logits, dim=1).item()

# Step 4: Convert number back to label
predicted_label = label_encoder.inverse_transform([predicted_class_id])
# Result: "Technical Support"
```

### Q20. What is `torch.no_grad()` and why is it used during inference?

**A:** `torch.no_grad()` tells PyTorch to **not compute gradients** during this computation.

- During **training**, gradients are needed for backpropagation (updating weights)
- During **inference** (prediction), we just want the output — no learning happens
- Skipping gradient computation **saves memory** and makes inference **faster**

### Q21. What is the keyword-based fallback in the classifier? Why is it needed?

**A:** Before using the BERT model, the classifier checks for **sales-related keywords** like "buy", "purchase", "pricing", "order", etc.

```python
if has_sales_intent and not has_problem:
    return "Sales and Pre-Sales"  # Skip BERT
```

Why: The BERT model is trained on support tickets, which are mostly **complaints**. When someone asks "I want to buy a laptop", the model might see "laptop" and classify it as Technical Support. The keyword filter catches obvious sales queries that the model struggles with.

This is a **hybrid approach**: rule-based for simple cases + ML for complex cases.

---

## Section 5: Backend Architecture

### Q22. Why did you choose FastAPI over Flask or Django?

**A:**
| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| Speed | Very fast (async) | Moderate | Moderate |
| Auto API docs | Yes (Swagger) | No | No |
| Type validation | Built-in (Pydantic) | Manual | Built-in |
| Learning curve | Easy | Easy | Steep |
| Async support | Native | Limited | Limited |

FastAPI gives us:

- **Automatic Swagger documentation** at `/docs`
- **Pydantic models** for request/response validation
- **Async endpoints** — important when calling the Gemini API (network I/O)
- **Type hints** — catches errors early

### Q23. Why SQLite for the database? What are its limitations?

**A:** SQLite was chosen because:

- **Zero setup** — No database server needed; it's a single file (`tickets.db`)
- **Sufficient for a prototype** — Handles thousands of tickets easily
- **Built into Python** — No additional installation required
- **Portable** — The database is just a file that can be copied

Limitations:

- **Not suitable for production** — No concurrent write support
- **No user connection** — Single-file, no network access
- **Replace with PostgreSQL /MySQL** for real deployment

### Q24. What ORM are you using and why?

**A:** **SQLAlchemy** — the most popular Python ORM (Object Relational Mapper).

It lets us write Python code instead of SQL:

```python
# Instead of: SELECT * FROM tickets WHERE client_id = 5
tickets = db.query(Ticket).filter(Ticket.client_id == 5).all()
```

Benefits:

- **Database-agnostic** — Switch from SQLite to PostgreSQL without changing code
- **Prevents SQL injection** — Automatically parameterizes queries
- **Pythonic** — Define tables as Python classes

### Q25. What is CORS and why is it configured in the project?

**A:** CORS (Cross-Origin Resource Sharing) is a browser security feature that blocks requests from one domain to another.

Our frontend (port 5173) and backend (port 8000) run on different ports = different origins. Without CORS configuration, the browser would block all API calls.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific origins in production)
    allow_methods=["*"],  # Allow GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)
```

### Q26. How does the department mapping work?

**A:** The ML model predicts one of 10 queues (e.g., "Technical Support", "IT Support"). These are mapped to 3 internal departments:

```python
DEPARTMENT_MAPPING = {
    "Technical Support": "technical_support",
    "IT Support": "technical_support",
    "Product Support": "technical_support",
    "Billing and Payments": "accounting",
    "Returns and Exchanges": "accounting",
    "Sales and Pre-Sales": "sales",
    "Customer Service": "sales",
    "General Inquiry": "sales",
}
```

This simplifies routing to 3 teams while keeping the ML classification granular.

---

## Section 6: Frontend Architecture

### Q27. What frontend technologies did you use?

**A:**

- **React 19** — Component-based UI library for building the interface
- **Vite 7** — Fast build tool and dev server (much faster than Webpack/Create React App)
- **Axios** — HTTP client for making API calls to the backend
- **Lucide React** — Icon library for clean UI icons
- **Vanilla CSS** — Custom styling without frameworks like Tailwind

### Q28. What are the main components of the frontend?

**A:**

1. **Login.jsx** — Login page with demo credentials
2. **App.jsx** — Main dashboard with three tabs:
   - **Create Ticket** — Form for submitting new tickets
   - **Ticket History** — Kanban board (Pending / In Progress / Resolved)
   - **Messages** — Shows the AI-generated reply
3. **TabsComponent.jsx** — Reusable tab navigation component

### Q29. How does the frontend handle different user roles?

**A:** The system has 5 roles with different views:

- **Client** — Can create tickets and see only their own tickets
- **Admin** — Can see ALL tickets from all departments
- **Technical Support / Accounting / Sales** — Can see only their department's tickets

The role is stored in `localStorage` after login and determines what data the API returns.

---

## Section 7: Docker & Deployment

### Q30. What is Docker and why did you use it?

**A:** Docker packages the application with all its dependencies into **containers** — isolated environments that run identically on any machine.

Why it's essential for this project:

- The project has complex dependencies: Python 3.11, PyTorch, Transformers, Node.js, and more
- Without Docker, setting up the environment takes **hours** and often fails on different OS versions
- With Docker: `docker compose up --build` → everything works in one command

### Q31. Explain the Docker architecture of this project.

**A:** We use **Docker Compose** with 2 services:

```
┌─────────────────────────────────────┐
│           Docker Compose            │
├──────────────────┬──────────────────┤
│    backend       │    frontend      │
│  (Python 3.11)   │  (Nginx + React) │
│  Port: 8000      │  Port: 5173      │
│                  │                  │
│  • FastAPI       │  • Nginx serves  │
│  • DistilBERT    │    built React   │
│  • Gemini API    │    static files  │
│  • SQLite        │  • Proxies /api/ │
│                  │    to backend    │
├──────────────────┴──────────────────┤
│  Volumes: model-data, db-data       │
└─────────────────────────────────────┘
```

### Q32. What is the purpose of Docker Volumes in this project?

**A:** Volumes persist data **across container restarts**:

- `model-data` → Stores the trained ML model (`/app/training/models/`) so it doesn't need retraining every time
- `db-data` → Stores the SQLite database (`/app/data/`) so tickets aren't lost

Without volumes, all data inside a container is **deleted** when the container stops.

### Q33. What is the Nginx reverse proxy doing?

**A:** The frontend Docker container uses Nginx to:

1. **Serve static files** — The built React app (HTML, CSS, JS)
2. **Proxy API calls** — Forward `/api/*` requests to the backend container

```nginx
location /api/ {
    proxy_pass http://backend:8000/api/;
}
```

This is essential in Codespaces/cloud where `localhost:8000` isn't accessible.

### Q34. What does the entrypoint script do?

**A:** `entrypoint.sh` runs when the backend container starts:

1. Checks if a trained model exists
2. If not → runs `train.py` to train the DistilBERT model (~3-5 min)
3. Starts the FastAPI server with Uvicorn
4. Waits for the server to be ready
5. Creates demo users via the `/api/init-users` endpoint

---

## Section 8: AI & NLP Concepts

### Q35. What is NLP? How is it used in this project?

**A:** NLP (Natural Language Processing) is a branch of AI that helps computers understand human language.

In this project:

- **Text Classification (NLP)** → Understanding what category a ticket belongs to
- **Text Generation (NLP)** → Writing a human-like auto-reply
- **Tokenization (NLP)** → Converting text to numbers the model can process

### Q36. What is a Transformer? Why is it important?

**A:** Transformer is the neural network architecture behind BERT, GPT, and all modern LLMs. Introduced in the 2017 paper "Attention is All You Need".

Key innovation: **Self-Attention Mechanism** — allows the model to look at ALL words in a sentence simultaneously (not one by one), understanding relationships between words regardless of their position.

Example: In "The **bank** of the **river**", attention helps the model understand "bank" means riverbank (not a financial bank) by attending to "river".

### Q37. What is Attention Mechanism?

**A:** Attention allows the model to focus on the **most relevant parts** of the input when making predictions.

For ticket classification:

- Input: "My **billing** statement shows a **wrong charge** from last month"
- The model pays more attention to "billing" and "wrong charge" than "from last month"
- This helps it correctly classify as "Billing and Payments"

### Q38. What is Fine-Tuning vs Training from Scratch?

**A:**
| Aspect | Training from Scratch | Fine-Tuning |
|--------|----------------------|-------------|
| Starting point | Random weights | Pre-trained weights |
| Data needed | Millions of samples | Hundreds to thousands |
| Time | Days/weeks | Minutes/hours |
| Compute needed | Multiple GPUs | Single CPU/GPU |
| Our approach | ❌ | ✅ |

We fine-tune because DistilBERT already knows English — we just teach it our specific categories.

### Q39. What is the difference between Classification and Generation models?

**A:**

- **Classification models** (like DistilBERT) → Input: text, Output: one label from a fixed set
  - Example: "My laptop is broken" → "Technical Support"
- **Generation models** (like Gemma 3) → Input: prompt, Output: free-form text
  - Example: prompt → "Dear customer, thank you for reaching out..."

Our project uses **both**: classification for routing, generation for replies.

### Q40. What is Prompt Engineering? How is it used?

**A:** Prompt engineering is the art of designing the input instruction given to an LLM to get the desired output.

In `gemini.py`, we use a structured prompt:

```
You are Shanyan AI Bot, a customer support assistant...
Ticket Number: TKT-20260222-ABCD
Customer Name: Rajesh Kumar
Customer Message: My laptop won't start...

Write a professional acknowledgement reply with this EXACT format:
Subject: ...
Dear {client_name},...
```

The prompt includes:

- **Role definition** ("You are Shanyan AI Bot")
- **Context** (ticket number, category, customer name)
- **Exact formatting instructions** (ensures consistent output)

---

## Section 9: Efficiency & Performance

### Q41. How efficient is the classification? What is the latency?

**A:**

- **DistilBERT inference**: ~50-100ms on CPU per ticket
- **Tokenization**: ~5ms
- **Total classification time**: Under 200ms

This is much faster than calling an LLM API (which takes 2-5 seconds). The customer gets their classification result almost instantly.

### Q42. How did you optimize the training for resource-constrained environments?

**A:** The training script automatically detects Docker/Codespace environments and adjusts:

| Setting          | Local Machine       | Docker/Codespace      |
| ---------------- | ------------------- | --------------------- |
| Training samples | Full dataset (~43K) | 500 samples           |
| Epochs           | 3                   | 1                     |
| Batch size       | 16                  | 8                     |
| Evaluation       | Per epoch           | Disabled              |
| Model saving     | Per epoch           | Final only            |
| Progress bars    | tqdm                | Custom print callback |

This reduces training time from **30+ minutes to ~3-5 minutes** on a 2-core CPU.

### Q43. What trade-offs did you make for Docker/cloud deployment?

**A:**

- **Accuracy vs Speed**: Training on 500 samples instead of 43K reduces accuracy slightly, but makes training feasible on limited hardware
- **Model size vs Quality**: DistilBERT (260MB) instead of full BERT (420MB) — 97% accuracy retained
- **Storage vs Retraining**: Docker volumes persist the trained model so it only trains once

### Q44. How scalable is this architecture?

**A:** Current limitations and improvements for production:

1. **SQLite → PostgreSQL** — for concurrent users
2. **Single container → Kubernetes** — for horizontal scaling
3. **Synchronous inference → Batch prediction** — for high throughput
4. **CPU inference → GPU** — for faster classification
5. **In-memory model → Model server (TorchServe)** — for better resource management
6. **Simple auth → JWT tokens** — for security

---

## Section 10: Libraries & Tools

### Q45. List all major libraries used and their purpose.

**A:**

**Backend (Python):**
| Library | Purpose |
|---------|---------|
| `FastAPI` | Web framework for REST API |
| `uvicorn` | ASGI server to run FastAPI |
| `transformers` | Hugging Face library for DistilBERT |
| `torch` (PyTorch) | Deep learning framework |
| `datasets` | Hugging Face datasets loader |
| `scikit-learn` | LabelEncoder, metrics (accuracy, F1) |
| `pandas` | Data manipulation (DataFrames) |
| `SQLAlchemy` | ORM for database operations |
| `Pydantic` | Data validation (request/response models) |
| `google-genai` | Gemini API client for Gemma 3 |
| `accelerate` | Optimizes Trainer for different hardware |
| `pickle` | Serialize/save the label encoder |

**Frontend (JavaScript):**
| Library | Purpose |
|---------|---------|
| `React` | UI component library |
| `Vite` | Build tool and dev server |
| `Axios` | HTTP client for API calls |
| `Lucide React` | Icon library |

### Q46. What is Hugging Face? Why is it used?

**A:** Hugging Face is the **GitHub of Machine Learning**. It provides:

- **Model Hub** — Pre-trained models (we download DistilBERT from here)
- **Datasets Hub** — Pre-built datasets (we load our training data from here)
- **Transformers library** — Code for loading, fine-tuning, and using models
- **Trainer API** — Simplified training loop with built-in optimizations

### Q47. What is PyTorch? Why use it over TensorFlow?

**A:** PyTorch is a deep learning framework by Meta.

Why PyTorch:

- **Hugging Face Transformers** is built on PyTorch (primary support)
- **Dynamic computation graphs** — easier to debug
- **Industry standard** — Most research uses PyTorch
- **Simpler API** — More Pythonic than TensorFlow

---

## Section 11: Security Considerations

### Q48. What are the security limitations of this project?

**A:** This is a prototype, so several security aspects are simplified:

1. **Passwords stored in plain text** — Should use bcrypt hashing
2. **Simple token auth** (`token_1_admin`) — Should use JWT (JSON Web Tokens)
3. **CORS allows all origins** (`"*"`) — Should restrict to specific domains
4. **No rate limiting** — Could be abused with many requests
5. **No input sanitization** — Vulnerable to injection attacks
6. **API key in .env file** — Should use a secrets manager in production

### Q49. How would you improve the authentication system?

**A:**

1. **Hash passwords** using `bcrypt` before storing in the database
2. **Use JWT tokens** with expiration time instead of simple strings
3. **Add refresh tokens** for session management
4. **Implement HTTPS** for encrypted communication
5. **Add rate limiting** to prevent brute-force attacks

---

## Section 12: Quick-Fire Technical Questions

### Q50. What does `model.eval()` do?

**A:** Switches the model from training mode to evaluation mode. This disables dropout layers and batch normalization updates, making predictions deterministic and correct.

### Q51. What is `pickle` used for?

**A:** `pickle` is Python's serialization library. We use it to save the LabelEncoder object to a file (`label_encoder.pkl`) so it can be loaded later without re-training.

### Q52. What is `softmax`?

**A:** A function that converts raw model output scores (logits) into probabilities that sum to 1. The class with the highest probability is the prediction.

### Q53. What is a REST API?

**A:** REST (Representational State Transfer) is an architectural style for APIs. It uses HTTP methods: GET (read), POST (create), PUT (update), DELETE (remove). Our API uses POST for login/predict and GET for fetching tickets.

### Q54. What is the purpose of `__init__` in Python?

**A:** The constructor method. Called when creating an instance of a class. In `TicketClassifierService.__init__()`, we load the model, tokenizer, and label encoder when the service starts.

### Q55. What is Pydantic?

**A:** A Python library for data validation using type hints. FastAPI uses it to automatically validate all incoming request data. For example, `TicketRequest(description: str, client_id: int)` ensures these fields exist and have the correct types.

### Q56. What is the difference between `save_pretrained()` and `pickle.dump()`?

**A:**

- `save_pretrained()` — Hugging Face method that saves model weights, config, and tokenizer in a specific format
- `pickle.dump()` — General Python serialization for any object. Used for the LabelEncoder because it's a scikit-learn object, not a Hugging Face model.

### Q57. What happens during `AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=10)`?

**A:** It downloads DistilBERT's pre-trained weights and adds a **classification head** — a new linear layer that maps from 768 hidden dimensions to 10 output categories. The pre-trained layers are kept; only the classification head starts with random weights.

---

## Tips for the Viva

1. **Draw the architecture diagram** when asked about the system overview
2. **Know the flow**: User → React → FastAPI → DistilBERT → Gemma 3 → Database → Response
3. **Key numbers to remember**: DistilBERT = 66M params, 6 layers, 97% of BERT accuracy. Gemma 3 = 27B params
4. **Be honest about limitations**: mention security concerns, SQLite limitations, accuracy trade-offs
5. **Practice explaining Transfer Learning** — it's the most likely deep question
6. **Know why two models**: Classification = small model (speed), Generation = large model (quality)
7. **Understand Docker basics**: containers, images, volumes, docker-compose

---

_Good luck, Shankar! 🍀_
