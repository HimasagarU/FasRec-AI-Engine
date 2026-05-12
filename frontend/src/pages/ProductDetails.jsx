import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';
import { Sparkles, Heart, BookmarkPlus, ArrowLeft, Tag, Palette, Calendar, Users } from 'lucide-react';

export default function ProductDetails() {
  const { id } = useParams();
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  const [narration, setNarration] = useState(null);
  const [loadingNarration, setLoadingNarration] = useState(false);
  
  const [isFavorite, setIsFavorite] = useState(false);
  const [savingOutfit, setSavingOutfit] = useState(false);
  const [outfitSaved, setOutfitSaved] = useState(false);

  useEffect(() => {
    const loadProduct = async () => {
      setLoading(true);
      setNarration(null);
      setOutfitSaved(false);
      try {
        const result = await api.getRecommendations(id);
        setData(result);
        
        // Check if already favorited
        if (user) {
          try {
            const favIds = await api.getFavoriteIds();
            setIsFavorite(parseInt(id) in favIds);
          } catch (e) { /* ignore */ }
        }
      } catch (err) {
        console.error("Failed to load product details", err);
      } finally {
        setLoading(false);
      }
    };
    loadProduct();
  }, [id, user]);

  const handleGenerateOutfit = async () => {
    setLoadingNarration(true);
    setOutfitSaved(false);
    try {
      const result = await api.getNarration(id);
      setNarration(result);
    } catch (err) {
      console.error("Failed to generate narration", err);
    } finally {
      setLoadingNarration(false);
    }
  };

  const handleToggleFavorite = async () => {
    if (!user) return alert("Please sign in to favorite items.");
    try {
      if (isFavorite) {
        await api.removeFavorite(parseInt(id));
        setIsFavorite(false);
      } else {
        await api.addFavorite(parseInt(id));
        setIsFavorite(true);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleSaveOutfit = async () => {
    if (!user) return alert("Please sign in to save outfits.");
    if (!narration || !data) return;
    
    setSavingOutfit(true);
    try {
      // Collect all product IDs from the outfit pieces
      const allProductIds = [];
      for (const piece of (narration.outfit_pieces || [])) {
        for (const p of (piece.products || [])) {
          allProductIds.push(p.id);
        }
      }
      await api.saveOutfit({
        query_product_id: parseInt(id),
        recommendation_text: narration.recommendation,
        occasion_text: narration.occasion,
        recommended_product_ids: allProductIds.slice(0, 20) // cap at 20
      });
      setOutfitSaved(true);
    } catch (err) {
      console.error(err);
    } finally {
      setSavingOutfit(false);
    }
  };

  if (loading) {
    return (
      <div className="container" style={{ textAlign: 'center', marginTop: '15vh' }}>
        <div className="spinner" style={{ margin: '0 auto', width: '40px', height: '40px' }}></div>
        <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>Loading product...</p>
      </div>
    );
  }

  if (!data) return <div className="container"><p>Product not found.</p><Link to="/discover" className="btn btn-secondary" style={{ marginTop: '1rem' }}><ArrowLeft size={16} /> Back to Discover</Link></div>;

  const { item, recommendations } = data;

  return (
    <div className="container">
      {/* Back Button */}
      <Link to="/discover" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
        <ArrowLeft size={16} /> Back to Discover
      </Link>

      {/* Product Hero */}
      <div className="product-detail-grid">
        <div className="glass-panel" style={{ padding: '1rem', position: 'sticky', top: '80px' }}>
          <img 
            src={api.getImageUrl(item.image_url)} 
            alt={item.title} 
            style={{ width: '100%', borderRadius: '12px', objectFit: 'contain', backgroundColor: '#1a1a20', maxHeight: '500px' }} 
          />
        </div>
        
        <div>
          <h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '1rem', lineHeight: 1.2 }}>{item.title}</h1>
          
          {/* Product Attributes */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div className="attr-chip"><Tag size={14} /> <span>{item.articleType}</span></div>
            <div className="attr-chip"><Palette size={14} /> <span>{item.baseColour}</span></div>
            <div className="attr-chip"><Users size={14} /> <span>{item.gender}</span></div>
            <div className="attr-chip"><Calendar size={14} /> <span>{item.season}</span></div>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: '1.5rem' }}>
            <span className="tag">{item.masterCategory}</span>
            <span className="tag">{item.subCategory}</span>
            <span className="tag">{item.usage}</span>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
            <button onClick={handleToggleFavorite} className={`btn ${isFavorite ? 'btn-favorited' : 'btn-secondary'}`}>
              <Heart size={18} fill={isFavorite ? '#ef4444' : 'none'} color={isFavorite ? '#ef4444' : 'currentColor'} />
              {isFavorite ? 'Favorited' : 'Add to Favorites'}
            </button>
            <button 
              onClick={handleGenerateOutfit} 
              className="btn btn-ai" 
              disabled={loadingNarration}
            >
              {loadingNarration ? (
                <><div className="spinner" style={{ width: '18px', height: '18px' }} /> Generating Outfit...</>
              ) : (
                <><Sparkles size={18} /> AI Outfit Generator</>
              )}
            </button>
          </div>

          {/* AI Outfit Result */}
          {narration && (
            <div className="glass-panel ai-section animate-fade-in">
              <div className="ai-header">
                <Sparkles size={22} />
                <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>AI Stylist Recommendation</h3>
              </div>
              
              <div className="ai-content">{narration.recommendation}</div>
              
              <div className="ai-occasion">
                <strong>🎯 Perfect for:</strong> {narration.occasion}
              </div>

              {/* Outfit Pieces with Real Products */}
              {narration.outfit_pieces && narration.outfit_pieces.length > 0 && (
                <div style={{ marginTop: '2rem' }}>
                  <h4 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', color: '#a78bfa' }}>
                    Complete Your Look
                  </h4>
                  {narration.outfit_pieces.map((piece, idx) => (
                    <div key={idx} className="outfit-piece animate-fade-in" style={{ animationDelay: `${idx * 0.1}s` }}>
                      <div className="outfit-piece-header">
                        <div>
                          <strong>{piece.type}</strong>
                          <span style={{ color: 'var(--text-secondary)', marginLeft: '0.5rem', fontSize: '0.85rem' }}>
                            in {piece.color}
                          </span>
                        </div>
                      </div>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', marginBottom: '0.75rem' }}>
                        {piece.why}
                      </p>
                      {piece.products && piece.products.length > 0 ? (
                        <div style={{ display: 'flex', gap: '0.75rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
                          {piece.products.map(p => (
                            <Link to={`/product/${p.id}`} key={p.id} className="outfit-product-card">
                              <img src={api.getImageUrl(p.image_url)} alt={p.title} />
                              <div className="outfit-product-info">
                                <span className="outfit-product-title">{p.title}</span>
                                <span className="outfit-product-meta">{p.baseColour}</span>
                              </div>
                            </Link>
                          ))}
                        </div>
                      ) : (
                        <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic', fontSize: '0.85rem' }}>
                          No exact matches found in catalog
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Save Outfit Button */}
              <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end' }}>
                {outfitSaved ? (
                  <span style={{ color: '#22c55e', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    ✓ Outfit Saved!
                  </span>
                ) : (
                  <button onClick={handleSaveOutfit} className="btn btn-secondary" disabled={savingOutfit}>
                    <BookmarkPlus size={18} /> {savingOutfit ? 'Saving...' : 'Save this Outfit'}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Similar Products */}
      {recommendations.length > 0 && (
        <section style={{ marginTop: '4rem' }}>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '1.5rem', borderBottom: '1px solid var(--border)', paddingBottom: '0.75rem' }}>
            Similar Products
          </h2>
          <div className="grid">
            {recommendations.slice(0, 10).map(p => (
              <Link to={`/product/${p.id}`} key={p.id} className="product-card animate-fade-in">
                <div className="product-image-container">
                  <img src={api.getImageUrl(p.image_url)} alt={p.title} className="product-image" loading="lazy" />
                  {p.score && (
                    <div className="score-badge">{(p.score * 100).toFixed(0)}% match</div>
                  )}
                </div>
                <div className="product-info">
                  <h3 className="product-title" title={p.title}>{p.title}</h3>
                  <div className="product-meta">{p.masterCategory} • {p.baseColour}</div>
                </div>
              </Link>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
