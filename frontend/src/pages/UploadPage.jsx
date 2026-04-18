import { useMemo, useState } from 'react'
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

  const canClassify = files.length > 0
  const recentJobs = useMemo(() => history.slice(0, 4), [history])

  const appendFiles = (fileList) => {
    const picked = Array.from(fileList).filter(
      (file) => file.type.startsWith('image/') || file.type === 'application/pdf',
    )
    if (!picked.length) return
    setFiles((prev) => [...prev, ...picked])
  }

  const runClassification = async () => {
    if (!canClassify) return
    const job = await createClassificationJob(files)
    setFiles([])
    navigate(`/analysis/${job.id}`)
  }

  return (
    <div className="page-shell">
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
              disabled={!canClassify}
              onClick={runClassification}
            >
              Classify Files
            </button>
          </article>

          <aside className="history-card">
            <div className="history-head">
              <h3>Recent Runs</h3>
              <span>{history.length} total</span>
            </div>
            {recentJobs.length === 0 && (
              <p className="empty-state">Your classification history will appear here.</p>
            )}
            {recentJobs.map((job) => (
              <Link key={job.id} className="history-item" to={`/analysis/${job.id}`}>
                <strong>{job.totalFiles} files</strong>
                <span>{job.invoicePages} invoice page(s) extracted</span>
              </Link>
            ))}
          </aside>
        </section>
      </main>
    </div>
  )
}

export default UploadPage
