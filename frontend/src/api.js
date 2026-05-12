const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const getAuthHeaders = () => {
  const token = localStorage.getItem('token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

const handleResponse = async (response) => {
  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user_email');
    }
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'API request failed');
  }
  return response.json();
};

export const api = {
  // Auth
  login: async (email, password) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    const res = await fetch(`${API_BASE_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData,
    });
    return handleResponse(res);
  },
  
  register: async (email, password) => {
    const res = await fetch(`${API_BASE_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    return handleResponse(res);
  },
  
  getMe: async () => {
    const res = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  // Products
  getProducts: async ({ page = 1, category = '', gender = '', search = '' }) => {
    const params = new URLSearchParams({ page });
    if (category) params.append('category', category);
    if (gender) params.append('gender', gender);
    if (search) params.append('search', search);
    const res = await fetch(`${API_BASE_URL}/products?${params}`);
    return handleResponse(res);
  },

  getCategories: async () => {
    const res = await fetch(`${API_BASE_URL}/categories`);
    return handleResponse(res);
  },

  getRecommendations: async (itemId) => {
    const res = await fetch(`${API_BASE_URL}/recommend/${itemId}`);
    return handleResponse(res);
  },

  getNarration: async (itemId) => {
    const res = await fetch(`${API_BASE_URL}/recommend/${itemId}/narration`);
    return handleResponse(res);
  },

  // Favorites
  getFavorites: async () => {
    const res = await fetch(`${API_BASE_URL}/favorites`, { headers: getAuthHeaders() });
    return handleResponse(res);
  },

  getFavoriteIds: async () => {
    const res = await fetch(`${API_BASE_URL}/favorites/ids`, { headers: getAuthHeaders() });
    return handleResponse(res);
  },

  addFavorite: async (productId) => {
    const res = await fetch(`${API_BASE_URL}/favorites`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
      body: JSON.stringify({ product_id: parseInt(productId) }),
    });
    return handleResponse(res);
  },

  removeFavorite: async (productId) => {
    const res = await fetch(`${API_BASE_URL}/favorites/${productId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },

  // Outfits
  getOutfits: async () => {
    const res = await fetch(`${API_BASE_URL}/outfits`, { headers: getAuthHeaders() });
    return handleResponse(res);
  },

  saveOutfit: async (data) => {
    const res = await fetch(`${API_BASE_URL}/outfits`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  removeOutfit: async (outfitId) => {
    const res = await fetch(`${API_BASE_URL}/outfits/${outfitId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });
    return handleResponse(res);
  },
  
  // Images
  getImageUrl: (urlPath) => {
    if (!urlPath) return '';
    if (urlPath.startsWith('http')) return urlPath;
    return `${API_BASE_URL}${urlPath}`;
  }
};
