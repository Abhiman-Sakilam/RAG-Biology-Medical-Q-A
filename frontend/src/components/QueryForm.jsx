import { useState } from 'react'
import './QueryForm.css'

export function QueryForm({ onSubmit, loading }) {
  const [value, setValue] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    onSubmit(value)
  }

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <label htmlFor="question" className="query-form__label">
        Ask a question
      </label>
      <textarea
        id="question"
        className="query-form__input"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="e.g. What is medulloblastoma? How is it treated?"
        rows={3}
        required
        disabled={loading}
        autoComplete="off"
      />
      <button
        type="submit"
        className="query-form__btn"
        disabled={loading || !value.trim()}
        aria-busy={loading}
      >
        {loading ? (
          <span className="query-form__spinner" aria-hidden />
        ) : (
          'Ask'
        )}
      </button>
    </form>
  )
}
