# Fashion Recommendation Engine

A high-performance, content-based fashion recommendation system that suggests similar products using multimodal AI (text + image). Built with **FastAPI**, **CLIP**, **SBERT**, and **FAISS**.

🌐 **Live Demo:** [https://fashion-recommendation-engine.onrender.com/app](https://fashion-recommendation-engine.onrender.com/app)

> **Note:** The demo is hosted on Render's free tier. It may take **30-60 seconds to wake up** after 15 minutes of inactivity. Please be patient! ⏳

## 🎥 Demo

![Demo Preview](https://via.placeholder.com/800x450.png?text=Add+Your+Demo+GIF+Here)

*Watch the full video walkthrough [here](#link-to-youtube-or-loom).*

## ✨ Features

*   **Multimodal Search**: Recommendations based on both visual similarity (CLIP) and semantic meaning (SBERT).
*   **Blazing Fast**: Uses **FAISS HNSW** index for millisecond-latency nearest neighbor search on 44k+ products.
*   **Smart Recommendations**: Hybrid fusion of text and image scores (`α * text_sim + (1-α) * image_sim`) for highly relevant results.
*   **Modern UI**: Dark-themed, responsive frontend with glassmorphism design.
*   **Cloud-Native**: Images served via **Cloudflare R2** CDN, app deployed on **Render**.

## 🛠️ Tech Stack

*   **Backend**: Python 3.11, FastAPI, Uvicorn
*   **AI/ML**: OpenAI CLIP (Vision), SBERT (Text), FAISS (Vector Search)
*   **Frontend**: HTML5, CSS3, Vanilla JS
*   **Infrastructure**: Docker, Render, Cloudflare R2

## 📦 Project Structure

```bash
├── artifacts/          # Precomputed embeddings & FAISS indexes
├── data/              # Dataset (styles.csv)
├── frontend/          # UI logic (index.html)
├── scripts/           # ML pipelines (embedding gen, indexing)
└── src/               # Application source code
```

## 🔧 Local Development

### Prerequisites
*   Python 3.11+
*   Docker (optional)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HimasagarU/Fashion-Recommendation-Engine.git
    cd Fashion-Recommendation-Engine
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python -m uvicorn src.api:app --reload
    ```
    Open http://localhost:8000/app in your browser.

### Using Docker

```bash
docker-compose up --build
```

## 📝 License

This project is open source and available under the MIT License.
