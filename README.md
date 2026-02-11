# Fashion Recommendation Engine

A content-based fashion recommendation system that suggests similar products using product details and image features. Built with K-means clustering and visual feature extraction techniques.

## 🌐 Live Demo

**Website:** [https://fashion-recommendation-engine.onrender.com/app](https://fashion-recommendation-engine.onrender.com/app)

## ✨ Features

- **Product Search**: Search for fashion items using the search bar
- **Smart Recommendations**: Click on any product to get personalized recommendations
- **Visual Similarity**: Recommendations based on both product details and image features
- **Diverse Catalog**: Browse through a wide range of fashion products across different categories and genders

## 🚀 How to Use

1. **Browse Products**: Scroll through the catalog to explore available fashion items
2. **Search**: Use the search bar at the top to find specific products
3. **Get Recommendations**: Click on any product card to view the top 10 similar items
4. **View Details**: Each product shows category, color, and gender information

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML/AI**: Sentence-BERT (SBERT), CLIP, K-means clustering, FAISS
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Docker, Render

## 📦 Project Structure

```
├── artifacts/          # Saved models and embeddings
├── data/              # Dataset files
├── frontend/          # Frontend HTML/CSS/JS
├── scripts/           # Data processing scripts
└── src/               # Backend source code
```

## 🔧 Local Development

### Prerequisites
- Python 3.8+
- Docker (optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/HimasagarU/Fashion-Recommendation-Engine.git
cd Fashion-Recommendation-Engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python src/app.py
```

### Using Docker

```bash
docker-compose up
```

## 📝 License

This project is open source and available under the MIT License.
