import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';
import { Heart, Bookmark, Trash2, Sparkles, ArrowRight, Package } from 'lucide-react';

export default function Profile() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('outfits');
  const [favorites, setFavorites] = useState([]);
  const [outfits, setOutfits] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;
    const loadData = async () => {
      setLoading(true);
      try {
        const [favs, outs] = await Promise.all([
          api.getFavorites(),
          api.getOutfits()
        ]);
        setFavorites(favs);
        setOutfits(outs);
      } catch (err) {
        console.error("Failed to load profile data", err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [user]);

  const removeFavorite = async (productId) => {
    try {
      await api.removeFavorite(productId);
      setFavorites(favorites.filter(f => f.id !== productId));
    } catch (err) {
      console.error(err);
    }
  };

  const removeOutfit = async (outId) => {
    try {
      await api.removeOutfit(outId);
      setOutfits(outfits.filter(o => o.id !== outId));
    } catch (err) {
      console.error(err);
    }
  };

  if (!user) {
    return (
      <div className="container" style={{ textAlign: 'center', paddingTop: '10vh' }}>
        <Sparkles size={48} color="var(--accent)" style={{ marginBottom: '1rem' }} />
        <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Sign in to view your collection</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>Save your favorite products and AI-generated outfits.</p>
        <Link to="/auth" className="btn btn-primary" style={{ padding: '0.75rem 2rem' }}>Sign In</Link>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="container" style={{ textAlign: 'center', marginTop: '10vh' }}>
        <div className="spinner" style={{ margin: '0 auto', width: '40px', height: '40px' }}></div>
      </div>
    );
  }

  return (
    <div className="container">
      {/* Profile Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '0.25rem' }}>My Collection</h1>
        <p style={{ color: 'var(--text-secondary)' }}>{user.email}</p>
      </div>

      {/* Stats */}
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
        <div className="glass-panel" style={{ padding: '1.25rem', flex: 1, textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: '#818cf8' }}>{outfits.length}</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Saved Outfits</div>
        </div>
        <div className="glass-panel" style={{ padding: '1.25rem', flex: 1, textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', fontWeight: 700, color: '#ef4444' }}>{favorites.length}</div>
          <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Favorited Items</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="profile-tabs">
        <button
          className={`profile-tab ${activeTab === 'outfits' ? 'active' : ''}`}
          onClick={() => setActiveTab('outfits')}
        >
          <Bookmark size={16} /> AI Outfits ({outfits.length})
        </button>
        <button
          className={`profile-tab ${activeTab === 'favorites' ? 'active' : ''}`}
          onClick={() => setActiveTab('favorites')}
        >
          <Heart size={16} /> Favorites ({favorites.length})
        </button>
      </div>

      {/* Outfits Tab */}
      {activeTab === 'outfits' && (
        <div>
          {outfits.length === 0 ? (
            <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
              <Sparkles size={40} color="var(--accent)" style={{ marginBottom: '1rem' }} />
              <h3 style={{ marginBottom: '0.5rem' }}>No saved outfits yet</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
                Go to any product and use the AI Outfit Generator to create outfits!
              </p>
              <Link to="/discover" className="btn btn-primary"><ArrowRight size={16} /> Discover Products</Link>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {outfits.map(o => (
                <div key={o.id} className="glass-panel animate-fade-in" style={{ padding: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                    <div style={{ flex: 1 }}>
                      <div className="ai-occasion" style={{ marginTop: 0, padding: '0.5rem 0.75rem', display: 'inline-block' }}>
                        🎯 {o.occasion_text}
                      </div>
                    </div>
                    <button onClick={() => removeOutfit(o.id)} className="btn btn-danger" style={{ padding: '0.4rem 0.6rem' }}>
                      <Trash2 size={14} />
                    </button>
                  </div>
                  
                  <p style={{ marginBottom: '1.25rem', fontSize: '0.95rem', lineHeight: 1.6, color: 'var(--text-secondary)' }}>
                    {o.recommendation_text}
                  </p>

                  <div style={{ display: 'flex', gap: '0.75rem', overflowX: 'auto', paddingBottom: '0.5rem' }}>
                    {/* Query Product */}
                    {o.query_product && (
                      <Link to={`/product/${o.query_product.id}`} className="outfit-product-card" style={{ border: '2px solid var(--accent)' }}>
                        <div style={{ fontSize: '0.7rem', background: 'var(--accent)', color: 'white', textAlign: 'center', padding: '0.2rem' }}>Main Item</div>
                        <img src={api.getImageUrl(o.query_product.image_url)} alt={o.query_product.title} />
                        <div className="outfit-product-info">
                          <span className="outfit-product-title">{o.query_product.title}</span>
                        </div>
                      </Link>
                    )}

                    {/* Recommended Products */}
                    {o.recommended_products.map((rp, idx) => (
                      <Link to={`/product/${rp.id}`} key={`${o.id}-${rp.id}-${idx}`} className="outfit-product-card">
                        <img src={api.getImageUrl(rp.image_url)} alt={rp.title} />
                        <div className="outfit-product-info">
                          <span className="outfit-product-title">{rp.title}</span>
                        </div>
                      </Link>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Favorites Tab */}
      {activeTab === 'favorites' && (
        <div>
          {favorites.length === 0 ? (
            <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
              <Heart size={40} color="#ef4444" style={{ marginBottom: '1rem' }} />
              <h3 style={{ marginBottom: '0.5rem' }}>No favorites yet</h3>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
                Click the heart icon on any product to save it here.
              </p>
              <Link to="/discover" className="btn btn-primary"><ArrowRight size={16} /> Discover Products</Link>
            </div>
          ) : (
            <div className="grid">
              {favorites.map(p => (
                <div key={p.id} className="product-card animate-fade-in" style={{ position: 'relative' }}>
                  <button
                    onClick={() => removeFavorite(p.id)}
                    className="card-remove-btn"
                  >
                    <Trash2 size={14} />
                  </button>
                  <Link to={`/product/${p.id}`} className="product-image-container">
                    <img src={api.getImageUrl(p.image_url)} alt={p.title} className="product-image" loading="lazy" />
                  </Link>
                  <div className="product-info">
                    <h3 className="product-title" title={p.title}>{p.title}</h3>
                    <div className="product-meta">{p.masterCategory} • {p.baseColour}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
