import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import * as XLSX from 'xlsx'
import AppHeader from '../components/AppHeader'
import { useAppState } from '../context/useAppState'

function showValue(value) {
  const v = (value ?? '').toString().trim()
  return v === '' ? '—' : v
}

function hasValue(value) {
  const v = (value ?? '').toString().trim()
  return v !== '' && v !== '—'
}

function buildExtractionRows(page, fieldBlobs) {
  const rows = [['Field', 'Value'], ['file_name', page.fileName]]
  fieldBlobs.forEach((field) => {
    rows.push([field.label, showValue(field.value)])
  })
  return rows
}

function toCsv(aoa) {
  return aoa
    .map((row) =>
      row
        .map((cell) => `"${String(cell ?? '').replaceAll('"', '""')}"`)
        .join(','),
    )
    .join('\n')
}

function AnalysisPage() {
  const { jobId } = useParams()
  const { getJobById } = useAppState()
  const job = getJobById(jobId)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [copiedField, setCopiedField] = useState('')
  const [showIssueModal, setShowIssueModal] = useState(false)
  const [issueSubmitted, setIssueSubmitted] = useState(false)
  const [isMisclassified, setIsMisclassified] = useState(false)
  const [trueLabel, setTrueLabel] = useState('')
  const [issueNotes, setIssueNotes] = useState('')
  const [missedFields, setMissedFields] = useState([{ field: '', value: '' }])

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
  const isInvoice = selectedPage.label === 'invoice'
  const fieldBlobs = selectedPage.extraction
    ? [
        { key: 'invoiceNumber', label: 'Invoice Number', value: selectedPage.extraction.invoiceNumber },
        { key: 'invoiceDate', label: 'Invoice Date', value: selectedPage.extraction.invoiceDate },
        { key: 'dueDate', label: 'Due Date', value: selectedPage.extraction.dueDate },
        { key: 'issuerName', label: 'Issuer Name', value: selectedPage.extraction.issuerName },
        { key: 'recipientName', label: 'Recipient Name', value: selectedPage.extraction.recipientName },
        { key: 'totalAmount', label: 'Total Amount', value: selectedPage.extraction.totalAmount },
      ].filter((item) => hasValue(item.value))
    : []
  const classificationBlobs = [
    { key: 'predictedClass', label: 'Predicted Class', value: selectedPage.label.toUpperCase() },
    { key: 'fileSize', label: 'File Size', value: `${(selectedPage.fileSize / 1024).toFixed(1)} KB` },
  ]

  const copyFieldValue = async (key, value) => {
    try {
      await navigator.clipboard.writeText(value)
      setCopiedField(key)
      window.setTimeout(() => setCopiedField(''), 1200)
    } catch {
      setCopiedField('')
    }
  }

  const resetIssueForm = () => {
    setIsMisclassified(false)
    setTrueLabel('')
    setIssueNotes('')
    setMissedFields([{ field: '', value: '' }])
    setIssueSubmitted(false)
  }

  const openIssueModal = () => {
    resetIssueForm()
    setShowIssueModal(true)
  }

  const closeIssueModal = () => {
    setShowIssueModal(false)
    setIssueSubmitted(false)
  }

  const updateMissedField = (index, key, nextValue) => {
    setMissedFields((prev) =>
      prev.map((entry, i) => (i === index ? { ...entry, [key]: nextValue } : entry)),
    )
  }

  const addMissedFieldRow = () => {
    setMissedFields((prev) => [...prev, { field: '', value: '' }])
  }

  const removeMissedFieldRow = (index) => {
    setMissedFields((prev) => {
      if (prev.length === 1) return prev
      return prev.filter((_, i) => i !== index)
    })
  }

  const submitIssue = (event) => {
    event.preventDefault()
    setIssueSubmitted(true)
  }

  const downloadCsv = () => {
    const rows = buildExtractionRows(selectedPage, fieldBlobs)
    const csv = toCsv(rows)
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${selectedPage.fileName.replace(/\.[^/.]+$/, '')}_extraction.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const downloadXlsx = () => {
    const rows = buildExtractionRows(selectedPage, fieldBlobs)
    const ws = XLSX.utils.aoa_to_sheet(rows)
    const wb = XLSX.utils.book_new()
    XLSX.utils.book_append_sheet(wb, ws, 'Extraction')
    XLSX.writeFile(wb, `${selectedPage.fileName.replace(/\.[^/.]+$/, '')}_extraction.xlsx`)
  }

  return (
    <div className="page-shell">
      {showIssueModal && (
        <div className="classify-overlay" role="dialog" aria-modal="true">
          <div className="classify-overlay-card issue-modal-card">
            {!issueSubmitted ? (
              <>
                <h3>Report an Issue</h3>
                <p>Help us improve this result by sharing what went wrong.</p>
                <form className="issue-form" onSubmit={submitIssue}>
                  <label className="issue-checkbox">
                    <input
                      type="checkbox"
                      checked={isMisclassified}
                      onChange={(e) => setIsMisclassified(e.target.checked)}
                    />
                    <span>This document was misclassified</span>
                  </label>

                  {isMisclassified && (
                    <label>
                      True label
                      <select value={trueLabel} onChange={(e) => setTrueLabel(e.target.value)}>
                        <option value="">Select true label</option>
                        <option value="invoice">Invoice</option>
                        <option value="form">Form</option>
                        <option value="resume">Resume</option>
                        <option value="email">Email</option>
                        <option value="budget">Budget</option>
                      </select>
                    </label>
                  )}

                  <div className="issue-section-head">
                    <strong>Missed fields and true values</strong>
                  </div>
                  {missedFields.map((entry, idx) => (
                    <div key={`mf-${idx}`} className="issue-row">
                      <select
                        value={entry.field}
                        onChange={(e) => updateMissedField(idx, 'field', e.target.value)}
                      >
                        <option value="">Field</option>
                        <option value="invoice_number">Invoice Number</option>
                        <option value="invoice_date">Invoice Date</option>
                        <option value="due_date">Due Date</option>
                        <option value="issuer_name">Issuer Name</option>
                        <option value="recipient_name">Recipient Name</option>
                        <option value="total_amount">Total Amount</option>
                      </select>
                      <input
                        type="text"
                        value={entry.value}
                        onChange={(e) => updateMissedField(idx, 'value', e.target.value)}
                        placeholder="True value"
                      />
                      <button
                        type="button"
                        className="copy-field-button"
                        onClick={() => removeMissedFieldRow(idx)}
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                  <button type="button" className="copy-field-button" onClick={addMissedFieldRow}>
                    Add field
                  </button>

                  <label>
                    Additional notes
                    <textarea
                      value={issueNotes}
                      onChange={(e) => setIssueNotes(e.target.value)}
                      rows={3}
                      placeholder="Optional context about the issue..."
                    />
                  </label>

                  <div className="issue-actions">
                    <button type="button" className="copy-field-button" onClick={closeIssueModal}>
                      Cancel
                    </button>
                    <button type="submit" className="auth-submit">
                      Submit report
                    </button>
                  </div>
                </form>
              </>
            ) : (
              <>
                <h3>Thanks, we got your report.</h3>
                <p>
                  We know extraction issues can be frustrating. Your feedback was recorded and
                  helps us improve model quality in future updates.
                </p>
                <div className="issue-actions">
                  <button type="button" className="auth-submit" onClick={closeIssueModal}>
                    Close
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
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
            <p className="predicted-class-display">{selectedPage.label}</p>

            <div className={isInvoice ? 'analysis-split' : 'analysis-split single'}>
              <div className="detail-box">
                <h4>Classification</h4>
                <div className="field-blob-grid">
                  {classificationBlobs.map((field) => (
                    <article key={field.key} className="field-blob">
                      <header>
                        <h5>{field.label}</h5>
                      </header>
                      <p>{showValue(field.value)}</p>
                    </article>
                  ))}
                </div>
                {selectedPage.backendError && (
                  <p className="meta-line">Backend note: {selectedPage.backendError}</p>
                )}
              </div>

              {isInvoice && (
                <div className="detail-box">
                  <div className="extractor-head">
                    <h4>Extractor Output</h4>
                  </div>

                  {selectedPage.extraction && fieldBlobs.length > 0 ? (
                    <div className="field-blob-grid">
                      {fieldBlobs.map((field) => (
                        <article key={field.key} className="field-blob">
                          <header>
                            <h5>{field.label}</h5>
                            <button
                              type="button"
                              className="copy-field-button"
                              onClick={() => copyFieldValue(field.key, field.value)}
                            >
                              {copiedField === field.key ? 'Copied' : 'Copy'}
                            </button>
                          </header>
                          <p>{showValue(field.value)}</p>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <p className="meta-line">
                      No non-empty extracted fields for this invoice.
                    </p>
                  )}

                  <div className="extractor-actions">
                    <button type="button" className="copy-field-button" onClick={downloadCsv}>
                      Download CSV
                    </button>
                    <button type="button" className="copy-field-button" onClick={downloadXlsx}>
                      Download XLSX
                    </button>
                  </div>
                </div>
              )}
            </div>
          </article>
        </section>

        <div className="analysis-actions">
          <button className="secondary-button" type="button" onClick={openIssueModal}>
            Report an Issue
          </button>
          <Link className="secondary-button" to="/app">
            Classify More Files
          </Link>
        </div>
      </main>
    </div>
  )
}

export default AnalysisPage
