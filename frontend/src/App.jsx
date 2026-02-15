import { useState } from 'react'
import { QueryForm } from './components/QueryForm'
import './App.css'
import { ResultCard } from './components/ResultCard'
import { Status } from './components/Status'
import { queryApi } from './api'

export default function App() {
  const [status, setStatus] = useState({ message: '', error: false })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(question) {
    if (!question?.trim()) return
    setLoading(true)
    setStatus({ message: 'Searching corpus and generating answer…', error: false })
    setResult(null)
    try {
      const data = await queryApi(question.trim())
      setResult(data)
      setStatus({ message: '', error: false })
    } catch (err) {
      setStatus({
        message: err.message || 'Request failed',
        error: true,
      })
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-logo">RAG</h1>
        <p className="app-tagline">
          Retrieval-augmented Q&amp;A over biology &amp; medical literature
        </p>
      </header>

      <main className="app-main">
        <QueryForm onSubmit={handleSubmit} loading={loading} />
        <Status message={status.message} error={status.error} />
        {result && (
          <ResultCard answer={result.answer} passages={result.passages} />
        )}
      </main>

      <footer className="app-footer">
        <p>Powered by BM25 retrieval and an LLM. Context from the corpus only.</p>
      </footer>
    </div>
  )
}
