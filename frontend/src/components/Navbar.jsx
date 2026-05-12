import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { LogOut, Heart, Sparkles, Search, User } from 'lucide-react';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navbar">
      <Link to="/" className="nav-brand">
        <Sparkles size={20} style={{ marginRight: '0.5rem' }} />
        FasRec AI
      </Link>
      
      <div className="nav-links">
        <Link to="/discover" className={`nav-link ${isActive('/discover') ? 'active' : ''}`}>
          <Search size={16} /> Discover
        </Link>
        {user ? (
          <>
            <Link to="/profile" className={`nav-link ${isActive('/profile') ? 'active' : ''}`} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Heart size={16} /> My Collection
            </Link>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{user.email}</span>
              <button onClick={handleLogout} className="btn btn-secondary" style={{ padding: '0.4rem 0.75rem', fontSize: '0.85rem' }}>
                <LogOut size={14} />
              </button>
            </div>
          </>
        ) : (
          <Link to="/auth" className="btn btn-primary" style={{ fontSize: '0.85rem' }}>
            <User size={16} /> Sign In
          </Link>
        )}
      </div>
    </nav>
  );
}
