import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import AppHeader from '../components/AppHeader'
import { useAppState } from '../context/useAppState'

function AnalysisPage() {
  const { jobId } = useParams()
  const { getJobById } = useAppState()
  const job = getJobById(jobId)
  const [selectedIndex, setSelectedIndex] = useState(0)

  if (!job) {
    return (
      <div className="page-shell">
        <AppHeader />
        <main className="workspace-main">
          <section className="workspace-header reveal-up delay-2">
            <h2>Analysis not found</h2>
            <p className="hero-description">
              This run might be unavailable. Start a new classification to generate results.
            </p>
            <Link className="primary-button" to="/app">
              Back to Upload
            </Link>
          </section>
        </main>
      </div>
    )
  }

  const selectedPage = job.pages[selectedIndex] ?? job.pages[0]

  return (
    <div className="page-shell">
      <AppHeader />

      <main className="workspace-main">
        <section className="workspace-header reveal-up delay-2">
          <p className="eyebrow">ANALYSIS RESULTS</p>
          <h2>Run summary for {job.totalFiles} uploaded file(s)</h2>
          <p className="hero-description">
            Select a file on the left to review the page and extracted fields.
          </p>
        </section>

        <section className="analysis-layout reveal-up delay-3">
          <aside className="analysis-card file-browser">
            <h3>Files in this run</h3>
            <div className="file-tabs">
              {job.pages.map((page, index) => (
                <button
                  key={`${page.fileName}-${index}`}
                  type="button"
                  className={index === selectedIndex ? 'file-tab active' : 'file-tab'}
                  onClick={() => setSelectedIndex(index)}
                >
                  {page.fileName}
                </button>
              ))}
            </div>

            <div className="preview-box">
              {selectedPage.previewUrl ? (
                <img src={selectedPage.previewUrl} alt={selectedPage.fileName} />
              ) : (
                <div className="preview-fallback">{selectedPage.fileName}</div>
              )}
            </div>
          </aside>

          <article className="analysis-card">
            <div className="analysis-head">
              <h3>{selectedPage.fileName}</h3>
              <span className="label-chip">{selectedPage.label}</span>
            </div>
            <p className="meta-line">Confidence: {(selectedPage.confidence * 100).toFixed(1)}%</p>

            <div className="analysis-split">
              <div className="detail-box">
                <h4>Classification</h4>
                <ul className="extraction-list">
                  <li>Predicted label: {selectedPage.label}</li>
                  <li>Confidence score: {(selectedPage.confidence * 100).toFixed(1)}%</li>
                  <li>File size: {(selectedPage.fileSize / 1024).toFixed(1)} KB</li>
                </ul>
              </div>

              <div className="detail-box">
                <h4>Extractor Output</h4>
                {selectedPage.extraction ? (
                  <ul className="extraction-list">
                    <li>Invoice number: {selectedPage.extraction.invoiceNumber}</li>
                    <li>Invoice date: {selectedPage.extraction.invoiceDate}</li>
                    <li>Due date: {selectedPage.extraction.dueDate}</li>
                    <li>Issuer name: {selectedPage.extraction.issuerName}</li>
                    <li>Recipient name: {selectedPage.extraction.recipientName}</li>
                    <li>Total amount: {selectedPage.extraction.totalAmount}</li>
                  </ul>
                ) : (
                  <p className="meta-line">
                    Extraction fields are available for pages labeled invoice.
                  </p>
                )}
              </div>
            </div>
          </article>
        </section>

        <div className="analysis-actions">
          <Link className="secondary-button" to="/app">
            Classify More Files
          </Link>
        </div>
      </main>
    </div>
  )
}

export default AnalysisPage
