import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../api';
import { Search, SlidersHorizontal } from 'lucide-react';

export default function Discovery() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [categories, setCategories] = useState({ masterCategories: [], genders: [] });
  const [selectedCat, setSelectedCat] = useState('');
  const [selectedGender, setSelectedGender] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  
  useEffect(() => {
    const init = async () => {
      try {
        const cats = await api.getCategories();
        setCategories(cats);
        await loadProducts(1, '', '', '');
      } catch (err) {
        console.error("Failed to load init data", err);
      }
    };
    init();
  }, []);

  const loadProducts = async (pageNum, cat, gender, searchTerm) => {
    setLoading(true);
    try {
      const data = await api.getProducts({ page: pageNum, category: cat, gender, search: searchTerm });
      if (pageNum === 1) {
        setProducts(data.products || []);
      } else {
        setProducts(prev => [...prev, ...(data.products || [])]);
      }
      setTotal(data.total || 0);
    } catch (err) {
      console.error("Failed to load products", err);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    setPage(1);
    loadProducts(1, selectedCat, selectedGender, search);
  };

  const handleCategoryChange = (cat) => {
    setSelectedCat(cat);
    setPage(1);
    loadProducts(1, cat, selectedGender, search);
  };

  const handleGenderChange = (gender) => {
    setSelectedGender(gender);
    setPage(1);
    loadProducts(1, selectedCat, gender, search);
  };

  const loadMore = () => {
    const newPage = page + 1;
    setPage(newPage);
    loadProducts(newPage, selectedCat, selectedGender, search);
  };

  const clearFilters = () => {
    setSelectedCat('');
    setSelectedGender('');
    setSearch('');
    setPage(1);
    loadProducts(1, '', '', '');
  };

  return (
    <div className="container">
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '2rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>Discover Fashion</h1>
        <p style={{ color: 'var(--text-secondary)' }}>{total.toLocaleString()} products available</p>
      </div>

      {/* Search & Filter Bar */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap', alignItems: 'stretch' }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', flexGrow: 1, maxWidth: '500px' }}>
          <input
            type="text"
            className="form-input"
            style={{ borderTopRightRadius: 0, borderBottomRightRadius: 0 }}
            placeholder="Search products..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button type="submit" className="btn btn-primary" style={{ borderTopLeftRadius: 0, borderBottomLeftRadius: 0, padding: '0.75rem 1rem' }}>
            <Search size={18} />
          </button>
        </form>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`btn ${showFilters ? 'btn-primary' : 'btn-secondary'}`}
        >
          <SlidersHorizontal size={16} /> Filters
        </button>
        {(selectedCat || selectedGender) && (
          <button onClick={clearFilters} className="btn btn-danger" style={{ fontSize: '0.85rem' }}>Clear All</button>
        )}
      </div>

      {/* Expandable Filters */}
      {showFilters && (
        <div className="glass-panel animate-fade-in" style={{ padding: '1.25rem', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', gap: '2rem', flexWrap: 'wrap' }}>
            <div>
              <label className="form-label">Category</label>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button onClick={() => handleCategoryChange('')} className={`tag ${!selectedCat ? 'tag-active' : ''}`}>All</button>
                {categories.masterCategories.map(c => (
                  <button key={c} onClick={() => handleCategoryChange(c)} className={`tag ${selectedCat === c ? 'tag-active' : ''}`}>{c}</button>
                ))}
              </div>
            </div>
            <div>
              <label className="form-label">Gender</label>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                <button onClick={() => handleGenderChange('')} className={`tag ${!selectedGender ? 'tag-active' : ''}`}>All</button>
                {categories.genders.map(g => (
                  <button key={g} onClick={() => handleGenderChange(g)} className={`tag ${selectedGender === g ? 'tag-active' : ''}`}>{g}</button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Product Grid */}
      {loading && products.length === 0 ? (
        <div className="grid">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="product-card skeleton" style={{ height: '350px' }}></div>
          ))}
        </div>
      ) : products.length === 0 ? (
        <div className="glass-panel" style={{ padding: '4rem', textAlign: 'center' }}>
          <p style={{ color: 'var(--text-secondary)', fontSize: '1.1rem' }}>No products found. Try adjusting your filters.</p>
        </div>
      ) : (
        <>
          <div className="grid">
            {products.map(p => (
              <Link to={`/product/${p.id}`} key={p.id} className="product-card animate-fade-in">
                <div className="product-image-container">
                  <img src={api.getImageUrl(p.image_url)} alt={p.title} className="product-image" loading="lazy" />
                </div>
                <div className="product-info">
                  <h3 className="product-title" title={p.title}>{p.title}</h3>
                  <div className="product-meta">{p.masterCategory} • {p.gender}</div>
                  <div>
                    <span className="tag">{p.baseColour}</span>
                    <span className="tag">{p.season}</span>
                  </div>
                </div>
              </Link>
            ))}
          </div>
          
          {products.length < total && (
            <div style={{ display: 'flex', justifyContent: 'center', marginTop: '3rem' }}>
              <button onClick={loadMore} className="btn btn-secondary" style={{ padding: '0.75rem 2.5rem' }}>
                Load More ({products.length} of {total.toLocaleString()})
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
