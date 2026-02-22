import React, { useState } from 'react';
import axios from 'axios';
import { LogIn, User, Lock, AlertCircle } from 'lucide-react';
import './Login.css';

const API_BASE = import.meta.env.VITE_API_URL || '';

function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_BASE}/api/login`, {
        username,
        password
      });
      
      // Store user data
      localStorage.setItem('user', JSON.stringify(response.data));
      onLogin(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div className="login-header">
          <div className="logo-circle">
            <User size={40} color="#6366f1" />
          </div>
          <h1>Shanyan AI</h1>
          <p>Enterprise Support Ticketing System</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label>
              <User size={18} />
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              required
              autoFocus
            />
          </div>

          <div className="form-group">
            <label>
              <Lock size={18} />
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
            />
          </div>

          {error && (
            <div className="error-message">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          <button type="submit" className="login-btn" disabled={loading}>
            {loading ? (
              'Signing in...'
            ) : (
              <>
                <LogIn size={20} />
                Sign In
              </>
            )}
          </button>
        </form>

        <div className="demo-accounts">
          <p><strong>Demo Accounts:</strong></p>
          <ul>
            <li><code>admin</code> / <code>admin123</code> - Administrator</li>
            <li><code>client1</code> / <code>client123</code> - Rajesh Kumar</li>
            <li><code>tech1</code> / <code>tech123</code> - Priya Sharma</li>
            <li><code>acc1</code> / <code>acc123</code> - Amit Patel</li>
            <li><code>sales1</code> / <code>sales123</code> - Sneha Reddy</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Login;
