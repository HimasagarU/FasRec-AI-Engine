import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Navbar from './components/Navbar';
import Landing from './pages/Landing';
import Discovery from './pages/Discovery';
import ProductDetails from './pages/ProductDetails';
import Auth from './pages/Auth';
import Profile from './pages/Profile';

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  if (loading) return <div className="container" style={{ textAlign: 'center', marginTop: '10vh' }}><div className="spinner" style={{ margin: '0 auto', width: '40px', height: '40px' }}></div></div>;
  if (!user) return <Navigate to="/auth" />;
  return children;
};

const AppContent = () => {
  return (
    <div className="app-container">
      <Navbar />
      <main style={{ flexGrow: 1 }}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/discover" element={<Discovery />} />
          <Route path="/product/:id" element={<ProductDetails />} />
          <Route path="/auth" element={<Auth />} />
          <Route path="/profile" element={
            <ProtectedRoute>
              <Profile />
            </ProtectedRoute>
          } />
        </Routes>
      </main>
      <footer style={{ borderTop: '1px solid var(--border)', padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
        <p>FasRec AI — GenAI Fashion Recommendation Engine</p>
        <p style={{ marginTop: '0.25rem', fontSize: '0.75rem' }}>CLIP + SBERT → FAISS HNSW → LLaMA 3.3-70B</p>
      </footer>
    </div>
  );
};

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
