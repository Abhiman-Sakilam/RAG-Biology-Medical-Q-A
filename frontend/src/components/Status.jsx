import './Status.css'

export function Status({ message, error }) {
  if (!message) return null
  return (
    <div
      className={`status ${error ? 'status--error' : ''}`}
      role="status"
      aria-live="polite"
    >
      {message}
    </div>
  )
}
