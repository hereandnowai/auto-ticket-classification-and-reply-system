import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Send, 
  Sparkles, 
  Loader2, 
  CheckCircle2, 
  MessageSquare, 
  ShieldCheck,
  Zap,
  LogOut,
  User,
  Ticket as TicketIcon,
  MessagesSquare,
  Clock,
  AlertCircle
} from 'lucide-react';
import Login from './Login';
import TabsComponent from './TabsComponent';
import './App.css';

// In Docker: empty string means requests go through nginx proxy (/api/ -> backend:8000)
// In local dev: set VITE_API_URL=http://localhost:8000 or it defaults to empty
const API_BASE = import.meta.env.VITE_API_URL || '';

function App() {
  const [user, setUser] = useState(null);
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('create');

  useEffect(() => {
    // Check for saved user session
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      const userData = JSON.parse(savedUser);
      setUser(userData);
      fetchHistory(userData.role, userData.user_id);
    }
  }, []);

  const fetchHistory = async (role, userId) => {
    try {
      const response = await axios.get(`${API_BASE}/api/tickets/${role}/${userId}`);
      setHistory(response.data);
    } catch (err) {
      console.error('Failed to fetch history');
    }
  };

  const handleLogin = (userData) => {
    setUser(userData);
    fetchHistory(userData.role, userData.user_id);
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    setUser(null);
    setHistory([]);
    setResult(null);
    setDescription('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!description.trim() || !user) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(`${API_BASE}/api/predict`, { 
        description: description,
        client_id: user.user_id,
        client_name: user.full_name
      });
      setResult(response.data);
      fetchHistory(user.role, user.user_id);
      setDescription('');
      setActiveTab('messages'); // Switch to messages tab to show response
    } catch (err) {
      setError(err.response?.data?.detail || 'System is currently recalibrating. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Show login if no user
  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  const canCreateTicket = user.role === 'client';
  const roleDisplay = user.role.replace('_', ' ').toUpperCase();

  return (
    <div className="app-wrapper">
      {/* Top Navigation Bar */}
      <nav className="top-nav">
        <div className="nav-left">
          <Sparkles size={24} color="#6366f1" />
          <span className="app-title">Shanyan AI Ticketing System</span>
        </div>
        <div className="nav-right">
          <div className="user-info-nav">
            <User size={18} />
            <span>{user.full_name}</span>
            <span className="role-badge-nav">{roleDisplay}</span>
          </div>
          <button onClick={handleLogout} className="logout-btn-nav">
            <LogOut size={18} />
          </button>
        </div>
      </nav>

      {/* Main Content Area */}
      <div className="main-content">
        <TabsComponent activeTab={activeTab} setActiveTab={setActiveTab}>
          {/* Tab 1: Create Ticket */}
          <div tabId="create" className="tab-content-panel">
            <div className="create-ticket-container">
              <h2 className="panel-title">
                <TicketIcon size={24} />
                Create New Ticket
              </h2>
              
              <form onSubmit={handleSubmit} className="ticket-form">
                <div className="form-group">
                  <label>Describe your issue</label>
                  <textarea
                    rows="10"
                    placeholder="Describe the issue in detail (e.g., 'I am unable to access my account dashboard since the last update...')"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    disabled={loading}
                    className="ticket-textarea"
                  />
                </div>

                <button type="submit" className="submit-btn" disabled={loading || !description.trim()}>
                  {loading ? (
                    <>
                      <Loader2 className="spinner" size={18} />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Send size={18} />
                      Submit Ticket
                    </>
                  )}
                </button>

                {error && (
                  <div className="error-message">
                    <AlertCircle size={18} />
                    {error}
                  </div>
                )}
              </form>
            </div>
          </div>

          {/* Tab 2: Tickets History with Kanban Board */}
          <div tabId="history" className="tab-content-panel">
            <h2 className="panel-title">
              <Clock size={20} />
              My Tickets ({history.length})
            </h2>
            
            <div className="kanban-board">
              {/* Pending Column */}
              <div className="kanban-column">
                <div className="column-header pending-header">
                  <span className="column-title">Pending</span>
                  <span className="column-count">
                    {history.filter(t => t.status === 'pending').length}
                  </span>
                </div>
                <div className="column-content">
                  {history.filter(t => t.status === 'pending').map(ticket => (
                    <div key={ticket.id} className="ticket-card">
                      <div className="card-header">
                        <span className="ticket-id">#{ticket.ticket_number}</span>
                        <span className="status-dot pending-dot"></span>
                      </div>
                      <h4 className="card-title">{ticket.predicted_queue}</h4>
                      <p className="card-body">{ticket.body.substring(0, 80)}...</p>
                      <div className="card-footer">
                        <span className="dept-tag">{ticket.assigned_department.replace('_', ' ')}</span>
                        <span className="time-label">
                          {new Date(ticket.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* In Progress Column */}
              <div className="kanban-column">
                <div className="column-header progress-header">
                  <span className="column-title">In Progress</span>
                  <span className="column-count">
                    {history.filter(t => t.status === 'in_progress').length}
                  </span>
                </div>
                <div className="column-content">
                  {history.filter(t => t.status === 'in_progress').map(ticket => (
                    <div key={ticket.id} className="ticket-card">
                      <div className="card-header">
                        <span className="ticket-id">#{ticket.ticket_number}</span>
                        <span className="status-dot progress-dot"></span>
                      </div>
                      <h4 className="card-title">{ticket.predicted_queue}</h4>
                      <p className="card-body">{ticket.body.substring(0, 80)}...</p>
                      <div className="card-footer">
                        <span className="dept-tag">{ticket.assigned_department.replace('_', ' ')}</span>
                        <span className="time-label">
                          {new Date(ticket.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Resolved Column */}
              <div className="kanban-column">
                <div className="column-header resolved-header">
                  <span className="column-title">Resolved</span>
                  <span className="column-count">
                    {history.filter(t => t.status === 'resolved').length}
                  </span>
                </div>
                <div className="column-content">
                  {history.filter(t => t.status === 'resolved').map(ticket => (
                    <div key={ticket.id} className="ticket-card">
                      <div className="card-header">
                        <span className="ticket-id">#{ticket.ticket_number}</span>
                        <span className="status-dot resolved-dot"></span>
                      </div>
                      <h4 className="card-title">{ticket.predicted_queue}</h4>
                      <p className="card-body">{ticket.body.substring(0, 80)}...</p>
                      <div className="card-footer">
                        <span className="dept-tag">{ticket.assigned_department.replace('_', ' ')}</span>
                        <span className="time-label">
                          {new Date(ticket.created_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Tab 3: Messages */}
          <div tabId="messages" className="tab-content-panel">
            <h2 className="panel-title">
              <MessagesSquare size={20} />
              Shanyan AI Messages
            </h2>
            
            {result ? (
              <div className="message-container">
                <div className="message-header">
                  <div className="ticket-badge">
                    <TicketIcon size={16} />
                    {result.ticket_number}
                  </div>
                  <div className="category-badge">
                    {result.queue}
                  </div>
                  <div className="dept-badge-msg">
                    {result.assigned_department.replace('_', ' ').toUpperCase()}
                  </div>
                </div>
                
                <div className="message-content">
                  <div className="ai-response">
                    <div className="ai-avatar">
                      <Sparkles size={20} color="#6366f1" />
                    </div>
                    <div className="message-bubble">
                      <div className="message-label">Shanyan AI Bot</div>
                      <div className="message-text">
                        {result.auto_reply}
                      </div>
                      <div className="message-time">
                        {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="empty-state">
                <MessagesSquare size={64} color="#e2e8f0" />
                <p>No messages yet</p>
                <p className="empty-hint">Submit a ticket to receive Shanyan AI-powered responses</p>
              </div>
            )}
          </div>
        </TabsComponent>
      </div>

      {/* Footer */}
      <footer className="app-footer">
        <p>This project is created by <strong>Shankar Narayanan</strong>, Student, Dr MGR University.</p>
        <p className="footer-tech">Powered by Google Gemma 3 27B & DistilBERT | © 2026 Shanyan AI</p>
      </footer>
    </div>
  );
}

export default App;
