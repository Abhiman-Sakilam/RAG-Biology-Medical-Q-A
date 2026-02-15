import { useState } from 'react'
import './ResultCard.css'

export function ResultCard({ answer, passages = [] }) {
  const [passagesOpen, setPassagesOpen] = useState(false)
  const hasAnswer = answer && answer.trim()

  return (
    <section className="result-card" aria-labelledby="result-title">
      <h2 id="result-title" className="result-card__title">
        Answer
      </h2>
      <div className="result-card__answer">
        {hasAnswer ? answer.trim() : 'No answer generated.'}
      </div>

      {passages.length > 0 && (
        <details
          className="result-card__passages"
          open={passagesOpen}
          onToggle={(e) => setPassagesOpen(e.target.open)}
        >
          <summary className="result-card__passages-summary">
            Retrieved passages ({passages.length})
          </summary>
          <div className="result-card__passages-list">
            {passages.map((p, i) => (
              <div key={p.passage_id ?? i} className="passage">
                <div className="passage__meta">
                  <span className="passage__badge">#{i + 1}</span>
                  <span>id {p.passage_id}</span>
                  <span className="passage__score">
                    score {typeof p.score === 'number' ? p.score.toFixed(3) : p.score}
                  </span>
                </div>
                <div className="passage__text">{p.passage || ''}</div>
              </div>
            ))}
          </div>
        </details>
      )}
    </section>
  )
}
