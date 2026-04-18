import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import AppHeader from '../components/AppHeader'
import { useAppState } from '../context/useAppState'

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function UploadPage() {
  const navigate = useNavigate()
  const { history, createClassificationJob } = useAppState()
  const [files, setFiles] = useState([])
  const [dragActive, setDragActive] = useState(false)
  const [isClassifying, setIsClassifying] = useState(false)
  const [loadingPhraseIndex, setLoadingPhraseIndex] = useState(0)
  const [expandedClass, setExpandedClass] = useState('')

  const loadingPhrases = [
    'Preparing document stream...',
    'Classifying pages...',
    'Extracting invoice information...',
    'Validating field consistency...',
    'Finalizing analysis output...',
  ]

  const canClassify = files.length > 0
  const classGroups = useMemo(() => {
    const labels = ['invoice', 'form', 'resume', 'email', 'budget']
    const grouped = Object.fromEntries(labels.map((label) => [label, []]))

    history.forEach((job) => {
      job.pages.forEach((page) => {
        const label = (page.label || '').toLowerCase()
        if (grouped[label]) {
          grouped[label].push(page.fileName)
        }
      })
    })

    labels.forEach((label) => {
      grouped[label] = grouped[label].slice(0, 6)
    })
    return grouped
  }, [history])

  const appendFiles = (fileList) => {
    const picked = Array.from(fileList).filter(
      (file) => file.type.startsWith('image/') || file.type === 'application/pdf',
    )
    if (!picked.length) return
    setFiles((prev) => [...prev, ...picked])
  }

  useEffect(() => {
    if (!isClassifying) return undefined
    const timer = window.setInterval(() => {
      setLoadingPhraseIndex((prev) => (prev + 1) % loadingPhrases.length)
    }, 1200)
    return () => window.clearInterval(timer)
  }, [isClassifying, loadingPhrases.length])

  const runClassification = async () => {
    if (!canClassify) return
    setIsClassifying(true)
    setLoadingPhraseIndex(0)
    try {
      const job = await createClassificationJob(files)
      setFiles([])
      navigate(`/analysis/${job.id}`)
    } finally {
      setIsClassifying(false)
    }
  }

  return (
    <div className="page-shell">
      {isClassifying && (
        <div className="classify-overlay" role="status" aria-live="polite">
          <div className="classify-overlay-card">
            <div className="loading-spinner" aria-hidden="true" />
            <h3>Running Extraction</h3>
            <p>{loadingPhrases[loadingPhraseIndex]}</p>
          </div>
        </div>
      )}
      <AppHeader />

      <main className="workspace-main">
        <section className="workspace-header reveal-up delay-2">
          <p className="eyebrow">CLASSIFIER WORKSPACE</p>
          <h2>Drop raw document images to classify with DiT flow</h2>
          <p className="hero-description">
            Supported input for this prototype: image files. Classification and
            invoice extraction run when you click Classify.
          </p>
        </section>

        <section className="upload-grid reveal-up delay-3">
          <article className="upload-card">
            <label
              className={dragActive ? 'dropzone active' : 'dropzone'}
              onDragOver={(event) => {
                event.preventDefault()
                setDragActive(true)
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(event) => {
                event.preventDefault()
                setDragActive(false)
                appendFiles(event.dataTransfer.files)
              }}
            >
              <input
                type="file"
                multiple
                accept="image/*,application/pdf"
                onChange={(event) => appendFiles(event.target.files)}
              />
              <p>Drag and drop document images here</p>
              <span>or click to browse files</span>
            </label>

            <div className="file-list">
              {files.length === 0 && <p className="empty-state">No files selected yet.</p>}
              {files.map((file, index) => (
                <div key={`${file.name}-${index}`} className="file-item">
                  <span>{file.name}</span>
                  <small>{formatSize(file.size)}</small>
                </div>
              ))}
            </div>

            <button
              className="auth-submit classify-button"
              type="button"
              disabled={!canClassify || isClassifying}
              onClick={runClassification}
            >
              {isClassifying ? 'Processing...' : 'Classify Files'}
            </button>
          </article>

          <aside className="history-card">
            <div className="history-head">
              <h3>Recent Runs</h3>
              <span>{history.length} total</span>
            </div>
            {history.length === 0 ? (
              <p className="empty-state">Your classification history will appear here.</p>
            ) : (
              <div className="class-group-grid">
                {Object.entries(classGroups).map(([label, fileNames]) => (
                  <button
                    key={label}
                    type="button"
                    className={expandedClass === label ? 'class-group-bubble active' : 'class-group-bubble'}
                    onClick={() => setExpandedClass((prev) => (prev === label ? '' : label))}
                  >
                    <strong>{label}</strong>
                    <span>{fileNames.length} file{fileNames.length === 1 ? '' : 's'}</span>
                    {expandedClass === label && (
                      <ul className="class-group-list">
                        {fileNames.length === 0 ? (
                          <li>No files in this class yet.</li>
                        ) : (
                          fileNames.map((name, idx) => <li key={`${label}-${idx}`}>{name}</li>)
                        )}
                      </ul>
                    )}
                  </button>
                ))}
              </div>
            )}
            {history[0] && (
              <Link className="history-item" to={`/analysis/${history[0].id}`}>
                <strong>Open latest run</strong>
                <span>View full page-by-page analysis</span>
              </Link>
            )}
          </aside>
        </section>
      </main>
    </div>
  )
}

export default UploadPage
