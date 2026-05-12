import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { Sparkles, Search, Zap, Brain, Shield } from 'lucide-react';

export default function Landing() {
  const [featured, setFeatured] = useState([]);

  useEffect(() => {
    const loadFeatured = async () => {
      try {
        const data = await api.getProducts({ page: 1 });
        setFeatured((data.products || []).slice(0, 8));
      } catch (err) {
        console.error(err);
      }
    };
    loadFeatured();
  }, []);

  return (
    <div>
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-glow" />
        <div className="container" style={{ position: 'relative', zIndex: 2, textAlign: 'center', paddingTop: '6rem', paddingBottom: '6rem' }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <span className="tag" style={{ background: 'rgba(99,102,241,0.2)', color: '#a78bfa', fontSize: '0.85rem', padding: '0.4rem 1rem' }}>
              Powered by LLaMA 3.3-70B × CLIP × FAISS
            </span>
          </div>
          <h1 style={{ fontSize: '3.5rem', fontWeight: 800, lineHeight: 1.1, marginBottom: '1.5rem', maxWidth: '700px', margin: '0 auto 1.5rem' }}>
            Your AI-Powered{' '}
            <span style={{ background: 'linear-gradient(to right, #818cf8, #c084fc, #f472b6)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Personal Stylist
            </span>
          </h1>
          <p style={{ fontSize: '1.2rem', color: 'var(--text-secondary)', maxWidth: '550px', margin: '0 auto 2.5rem', lineHeight: 1.6 }}>
            Discover 44,000+ fashion products. Get AI-curated outfit recommendations with real product matches from our catalog.
          </p>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link to="/discover" className="btn btn-primary" style={{ padding: '0.75rem 2rem', fontSize: '1.05rem', background: 'linear-gradient(to right, #6366f1, #8b5cf6)' }}>
              <Search size={20} /> Explore Collection
            </Link>
            <Link to="/auth" className="btn btn-secondary" style={{ padding: '0.75rem 2rem', fontSize: '1.05rem' }}>
              <Sparkles size={20} /> Get Started Free
            </Link>
          </div>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="container" style={{ paddingTop: '4rem', paddingBottom: '4rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(99,102,241,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
              <Brain size={24} color="#818cf8" />
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '0.5rem' }}>GenAI Outfit Builder</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
              LLaMA 3.3-70B analyzes your selected item and builds a complete outfit with real matching products from 44k+ items.
            </p>
          </div>
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(167,139,250,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
              <Zap size={24} color="#a78bfa" />
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '0.5rem' }}>Sub-100ms Retrieval</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
              CLIP + SBERT multimodal embeddings fused into FAISS HNSW index for lightning-fast similarity search.
            </p>
          </div>
          <div className="glass-panel" style={{ padding: '2rem' }}>
            <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(244,114,182,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem' }}>
              <Shield size={24} color="#f472b6" />
            </div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 600, marginBottom: '0.5rem' }}>Save & Collect</h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: 1.6 }}>
              Favorite products and save AI-generated outfits to your personal collection. Access them anytime.
            </p>
          </div>
        </div>
      </section>

      {/* Featured Products Preview */}
      {featured.length > 0 && (
        <section className="container" style={{ paddingBottom: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
            <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.5rem' }}>Trending Now</h2>
            <p style={{ color: 'var(--text-secondary)' }}>Explore from our collection of 44,000+ products</p>
          </div>
          <div className="grid">
            {featured.map(p => (
              <Link to={`/product/${p.id}`} key={p.id} className="product-card animate-fade-in">
                <div className="product-image-container">
                  <img src={api.getImageUrl(p.image_url)} alt={p.title} className="product-image" loading="lazy" />
                </div>
                <div className="product-info">
                  <h3 className="product-title" title={p.title}>{p.title}</h3>
                  <div className="product-meta">{p.masterCategory} • {p.gender}</div>
                  <div><span className="tag">{p.baseColour}</span></div>
                </div>
              </Link>
            ))}
          </div>
          <div style={{ textAlign: 'center', marginTop: '2.5rem' }}>
            <Link to="/discover" className="btn btn-secondary" style={{ padding: '0.75rem 2rem' }}>
              View All Products →
            </Link>
          </div>
        </section>
      )}
    </div>
  );
}
