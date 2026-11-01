
import { useEffect, useState } from "react";
import "./App.css";

// Local development uses Flask on port 5000.
// For deployment, configure VITE_API_URL in Render.
const API = `${import.meta.env.VITE_API_URL || "http://127.0.0.1:5000"}/api`;

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Load previous predictions from Flask.
  useEffect(() => {
    async function loadHistory() {
      try {
        const response = await fetch(`${API}/history`);
        if (!response.ok) return;

        const data = await response.json();
        setHistory(data);
      } catch {
        console.log("History will load when the backend is available.");
      }
    }

    loadHistory();
  }, []);

  async function analyzeComment(event) {
    event.preventDefault();
    setError("");
    setResult(null);

    if (!text.trim()) {
      setError("Please enter a comment first.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text.trim() }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      setResult(data);

      setHistory((previous) => [
        data,
        ...previous.filter((item) => item.id !== data.id),
      ].slice(0, 50));

    } catch (err) {
      setError(
        `${err.message} Check whether your Flask backend is running.`
      );
    } finally {
      setLoading(false);
    }
  }

  function clearForm() {
    setText("");
    setResult(null);
    setError("");
  }

  const toxicCount = history.filter((item) => item.label === 1).length;
  const safeCount = history.length - toxicCount;

  return (
    <main className="app-shell">

      {/* NAVIGATION */}

      <nav className="navbar">
        <a className="brand" href="#home">
          <span className="brand-icon">✦</span>
          CyberShield <span className="brand-ai">AI</span>
        </a>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#image-detection">Image Detection</a>
          <a href="#analytics">Analytics</a>
          <a href="#history">History</a>
          <a href="#about">About</a>
        </div>

        <span className="nav-status">
          <span className="live-dot"></span>
          AI Safety Lab
        </span>
      </nav>

      {/* HERO */}

      <section className="hero" id="home">
        <div className="hero-content">
          <span className="eyebrow">
            ✦ MULTILINGUAL AI · ONLINE SAFETY
          </span>

          <h1>
            A safer internet
            <br />
            starts <span>with us.</span>
          </h1>

          <p>
            Detect potentially harmful online comments using
            machine learning. Making digital spaces safer,
            one conversation at a time.
          </p>

          <div className="hero-tags">
            <span>English</span>
            <span>తెలుగు</span>
            <span>Hinglish</span>
          </div>

          <a href="#analyze" className="hero-btn">
            Start analyzing →
          </a>
        </div>

        <div className="hero-visual">
          <div className="visual-circle">
            <div className="visual-shield">🛡️</div>
            <span className="orbit orbit-one">AI</span>
            <span className="orbit orbit-two">NLP</span>
            <span className="orbit orbit-three">SAFE</span>
          </div>
          <p>Technology for kinder conversations.</p>
        </div>
      </section>

      {/* FEATURE CARDS */}

      <section className="feature-grid">
        <div className="feature-card">
          <div className="feature-icon purple">✎</div>
          <h3>Text Analysis</h3>
          <p>Analyze comments for potentially harmful language.</p>
          <span className="feature-status available">Available</span>
        </div>

        <div className="feature-card">
          <div className="feature-icon blue">▧</div>
          <h3>Image Detection</h3>
          <p>Future image-based cyberbullying detection workspace.</p>
          <span className="feature-status upcoming">Coming soon</span>
        </div>

        <div className="feature-card">
          <div className="feature-icon green">◈</div>
          <h3>AI Classification</h3>
          <p>Machine learning models for toxicity classification.</p>
          <span className="feature-status available">Available</span>
        </div>

        <div className="feature-card">
          <div className="feature-icon orange">◷</div>
          <h3>Analysis History</h3>
          <p>Review your previously analyzed comments.</p>
          <span className="feature-status available">Available</span>
        </div>
      </section>

      {/* TEXT ANALYSIS */}

      <section className="section-heading" id="analyze">
        <span className="eyebrow">01 / TEXT MODERATION</span>
        <h2>Let's analyze a comment.</h2>
        <p>Enter a message and let the model examine its language patterns.</p>
      </section>

      <section className="analysis-layout">
        <div className="panel analysis-panel">
          <div className="panel-top">
            <div>
              <span className="mini-label">COMMENT ANALYZER</span>
              <h3>Check a comment</h3>
            </div>
            <span className="status-badge">
              <span className="live-dot"></span> Ready
            </span>
          </div>

          <form onSubmit={analyzeComment}>
            <label htmlFor="comment">Your comment</label>

            <textarea
              id="comment"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Type or paste a comment here..."
              maxLength={2000}
              rows={7}
            />

            <div className="input-meta">
              <span>English · తెలుగు · Hinglish</span>
              <span>{text.length}/2000</span>
            </div>

            <div className="button-row">
              <button className="primary-btn" disabled={loading}>
                {loading ? "Analyzing..." : "Analyze comment →"}
              </button>

              <button
                type="button"
                className="secondary-btn"
                onClick={clearForm}
              >
                Clear
              </button>
            </div>
          </form>

          {error && <p className="error-message">{error}</p>}

          {result && (
            <div
              className={`result-card ${
                result.label === 1 ? "flagged" : "safe"
              }`}
              aria-live="polite"
            >
              <span className="mini-label">ANALYSIS RESULT</span>

              <h3>{result.label === 1 ? "⚠ Potentially harmful" : "✓ Likely non-bullying"}</h3>

              <p>
                Model score: {(result.score * 100).toFixed(1)}%
              </p>

              {result.categories &&
                Object.keys(result.categories).length > 0 && (
                  <div className="category-list">
                    <strong>Detected categories</strong>
                    <div className="category-tags">
                      {Object.entries(result.categories).map(
                        ([category, detected]) => (
                          <span
                            key={category}
                            className={detected ? "category-active" : "category-inactive"}
                          >
                            {category.replace("_", " ")}
                          </span>
                        )
                      )}
                    </div>
                  </div>
                )}

              <p className="result-warning">{result.warning}</p>
              <p className="small-note">
                Experimental model score, not a verified probability.
                Human review is recommended.
              </p>
            </div>
          )}
        </div>

        <aside className="panel explanation-panel">
          <span className="mini-label">HOW IT WORKS</span>
          <h3>Words matter.</h3>
          <p className="muted">
            Our experimental system uses natural language processing
            to identify potentially harmful comments.
          </p>

          <div className="info-item">
            <span>01</span>
            <div>
              <strong>Text patterns</strong>
              <p>Learn patterns from labeled examples.</p>
            </div>
          </div>

          <div className="info-item">
            <span>02</span>
            <div>
              <strong>Language challenges</strong>
              <p>Mixed scripts and transliteration need careful testing.</p>
            </div>
          </div>

          <div className="info-item">
            <span>03</span>
            <div>
              <strong>Human judgment</strong>
              <p>Assist moderation, not automatic punishment.</p>
            </div>
          </div>
        </aside>
      </section>

      {/* FUTURE IMAGE DETECTION SECTION */}

      <section className="section-heading" id="image-detection">
        <span className="eyebrow">02 / MULTIMODAL MODERATION</span>
        <h2>Image detection workspace.</h2>
        <p>
          Our next step is to extend cyberbullying detection
          beyond text and into image-based content.
        </p>
      </section>

      <section className="image-workspace panel">
        <div className="image-icon">▧</div>
        <span className="feature-status upcoming">Coming soon</span>

        <h3>Image-based cyberbullying detection</h3>

        <p>
          Uploading images will be supported in a future version.
          We plan to explore OCR, image analysis and text-image
          context to identify potentially harmful content.
        </p>

        <div className="upload-placeholder">
          <div className="upload-symbol">↑</div>
          <strong>Image upload area</strong>
          <span>PNG · JPG · JPEG · WEBP</span>
          <button type="button" disabled>
            Image analysis coming soon
          </button>
        </div>

        <div className="planned-features">
          <span>✦ Image upload</span>
          <span>✦ Text extraction (OCR)</span>
          <span>✦ Visual content analysis</span>
          <span>✦ Multimodal classification</span>
        </div>
      </section>

      {/* ANALYTICS */}

      <section className="section-heading" id="analytics">
        <span className="eyebrow">03 / ANALYTICS</span>
        <h2>Your analysis overview.</h2>
        <p>Statistics from the prediction history currently available.</p>
      </section>

      <section className="stats-grid">
        <div className="stat-card">
          <span>Total analyzed</span>
          <strong>{history.length}</strong>
          <small>Stored predictions loaded</small>
        </div>

        <div className="stat-card">
          <span>Potentially harmful</span>
          <strong>{toxicCount}</strong>
          <small>Flagged predictions</small>
        </div>

        <div className="stat-card">
          <span>Likely non-bullying</span>
          <strong>{safeCount}</strong>
          <small>Non-flagged predictions</small>
        </div>
      </section>

      {/* HISTORY */}

      <section className="section-heading" id="history">
        <span className="eyebrow">04 / HISTORY</span>
        <h2>Recent analyses.</h2>
        <p>Your latest prediction records from the backend.</p>
      </section>

      <section className="panel history-panel">
        <div className="panel-top">
          <h3>Prediction history</h3>
          <span className="history-count">{history.length} records</span>
        </div>

        {history.length === 0 ? (
          <p className="empty-state">
            No analysis history yet. Try analyzing your first comment!
          </p>
        ) : (
          <div className="history-list">
            {history.map((item, index) => (
              <div className="history-item" key={item.id ?? index}>
                <div className="history-text">
                  <p>{item.text}</p>
                  <small>
                    {item.created_at
                      ? new Date(item.created_at).toLocaleString()
                      : "Recent session"}
                  </small>
                </div>

                <span className={item.label === 1 ? "pill danger" : "pill neutral"}>
                  {item.label === 1 ? "Potentially harmful" : "Likely non-bullying"}
                </span>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ABOUT */}

      <section className="about-section" id="about">
        <span className="eyebrow">05 / ABOUT THE PROJECT</span>
        <h2>Technology with a purpose.</h2>
        <p>
          CyberShield AI is an educational project exploring
          machine learning and multilingual cyberbullying detection.
          It aims to support safer and more respectful online communities.
        </p>

        <div className="tech-tags">
          <span>React</span>
          <span>Flask</span>
          <span>Python</span>
          <span>Scikit-learn</span>
          <span>TF-IDF</span>
          <span>SQLite</span>
        </div>
      </section>

      {/* FOOTER */}

      <footer className="footer">
        <a className="brand" href="#home">
          <span className="brand-icon">✦</span>
          CyberShield AI
        </a>

        <p>Built for safer digital conversations.</p>
        <small>
          Educational prototype · Predictions require human review.
        </small>
      </footer>

    </main>
  );
}

export default App;
