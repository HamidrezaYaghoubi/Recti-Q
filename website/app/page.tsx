const results = [
  {
    value: "+7.37",
    unit: "pp",
    label: "Recovery on impulse noise",
    detail: "ResNet-50 · ImageNet-C · severity 5",
  },
  {
    value: "99%+",
    unit: "",
    label: "PTQ memory savings retained",
    detail: "A robustness patch without the full model",
  },
  {
    value: "6",
    unit: "KB",
    label: "Smallest effective adapter",
    detail: "Rank 4 recovers 74% of the maximum gain",
  },
];

const pipeline = [
  {
    number: "01",
    title: "Quantize",
    body: "Apply calibration-free, weight-only 4-bit PTQ to a pretrained vision model.",
  },
  {
    number: "02",
    title: "Freeze",
    body: "Keep the entire efficient backbone fixed. No expensive end-to-end retraining.",
  },
  {
    number: "03",
    title: "Rectify",
    body: "Train one low-rank adapter on source data to correct pre-classifier features.",
  },
  {
    number: "04",
    title: "Deploy",
    body: "Send only the tiny adapter as an over-the-air resilience patch to the edge.",
  },
];

const pacsRows = [
  ["DeiT-s", "Sketch", "68.59", "64.57", "65.74", "+1.17"],
  ["DeiT-t", "Cartoon", "73.08", "71.72", "73.34", "+1.62"],
  ["DeiT-t", "Art painting", "74.90", "73.63", "75.15", "+1.52"],
  ["ResNet-50", "Sketch", "72.46", "72.42", "73.30", "+0.88"],
];

function ArrowIcon() {
  return <span aria-hidden="true">↗</span>;
}

export default function Home() {
  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Recti-Q home">
          <span className="brand-mark">R/Q</span>
          <span>Recti-Q</span>
        </a>
        <nav aria-label="Main navigation">
          <a href="#method">Method</a>
          <a href="#results">Results</a>
          <a href="#deployment">Deployment</a>
        </nav>
        <a className="header-link" href="/recti-q-paper.pdf" target="_blank">
          Read paper <ArrowIcon />
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-grid" aria-hidden="true" />
        <div className="hero-copy">
          <div className="eyebrow-row">
            <span className="eyebrow">IROS 2026</span>
            <span className="eyebrow-note">Edge robotics · Robust perception</span>
          </div>
          <h1>
            Robust 4-bit perception,
            <span> repaired in feature space.</span>
          </h1>
          <p className="hero-summary">
            Recti-Q closes the hidden robustness gap introduced by post-training
            quantization—with a tiny source-only adapter and no changes to the
            frozen backbone.
          </p>
          <div className="hero-actions">
            <a className="button button-primary" href="/recti-q-paper.pdf" target="_blank">
              Read the paper <ArrowIcon />
            </a>
            <a
              className="button button-secondary"
              href="https://github.com/HamidrezaYaghoubi/Recti-Q"
              target="_blank"
              rel="noreferrer"
            >
              View code <ArrowIcon />
            </a>
          </div>
          <div className="authors">
            <p>Hamidreza Yaghoubi Araghi<sup>*</sup></p>
            <p>Parastoo Pilevar<sup>*</sup></p>
            <p>Ming C. Lin</p>
            <span>University of Maryland · <sup>*</sup>Equal contribution</span>
          </div>
        </div>

        <div className="hero-visual" aria-label="Recti-Q concept illustration">
          <div className="visual-kicker">
            <span>Quantized backbone</span>
            <span className="status-dot">Frozen</span>
          </div>
          <div className="signal signal-noisy" />
          <div className="backbone-stack">
            <span />
            <span />
            <span />
            <span />
          </div>
          <div className="adapter-block">
            <small>LoRA</small>
            <strong>Δ</strong>
            <span>&lt;1% params</span>
          </div>
          <div className="signal signal-clean" />
          <div className="output-target">
            <span />
            <span />
            <span />
          </div>
          <p className="visual-caption">Feature-space correction · source data only</p>
        </div>
      </section>

      <section className="problem section-shell">
        <div className="section-label">The problem</div>
        <div className="problem-copy">
          <h2>Clean accuracy says “ready.”<br />The real world says otherwise.</h2>
          <p>
            Four-bit PTQ can preserve in-distribution accuracy while silently
            weakening a model under sensor noise, severe weather, and unfamiliar
            environments. We call this the <strong>Quantization-Induced Robustness Gap.</strong>
          </p>
        </div>
        <div className="gap-card">
          <div className="gap-card-head">
            <span>DeiT-S · Contrast corruption</span>
            <span>ImageNet-C · Severity 5</span>
          </div>
          <div className="bar-row">
            <span>FP32</span>
            <div className="bar-track"><i style={{ width: "100%" }} /></div>
            <strong>39.46</strong>
          </div>
          <div className="bar-row danger">
            <span>W4</span>
            <div className="bar-track"><i style={{ width: "85.8%" }} /></div>
            <strong>33.84</strong>
          </div>
          <div className="gap-loss"><b>−5.62 pp</b> hidden OOD drop</div>
        </div>
      </section>

      <section className="method section-shell" id="method">
        <div className="section-heading">
          <div>
            <div className="section-label">The method</div>
            <h2>Repair the feature space.<br />Leave the backbone alone.</h2>
          </div>
          <p>
            Recti-Q learns a low-rank update at the classifier head. The adapter
            observes pre-classifier features and adds a small logit correction,
            while the quantized backbone stays frozen.
          </p>
        </div>

        <div className="equation-card">
          <div>
            <span className="equation-label">Rectified prediction</span>
            <p>z = z<sub>q</sub> + B(A(u)) · α/r</p>
          </div>
          <div className="equation-notes">
            <span><b>u</b> pre-classifier features</span>
            <span><b>A, B</b> low-rank projections</span>
            <span><b>z<sub>q</sub></b> frozen W4 logits</span>
          </div>
        </div>

        <div className="method-figure">
          <img
            src="/assets/recti-q-method.png"
            alt="Recti-Q training and inference pipeline with frozen teacher and quantized student, and a trainable LoRA adapter"
          />
          <p>
            The optional FP32 teacher is used only during training. Recti-Q also
            supports a teacher-free mode for maximum efficiency.
          </p>
        </div>

        <div className="pipeline-grid">
          {pipeline.map((item) => (
            <article key={item.number}>
              <span>{item.number}</span>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="results" id="results">
        <div className="section-shell">
          <div className="results-intro">
            <div>
              <div className="section-label light">Results</div>
              <h2>Tiny patch.<br />Measurable recovery.</h2>
            </div>
            <p>
              Across CNNs and Transformers, Recti-Q improves robustness under
              domain shifts and common corruptions while preserving the memory
              advantage of W4 deployment.
            </p>
          </div>

          <div className="stat-grid">
            {results.map((result) => (
              <article key={result.label}>
                <div><strong>{result.value}</strong><span>{result.unit}</span></div>
                <h3>{result.label}</h3>
                <p>{result.detail}</p>
              </article>
            ))}
          </div>

          <div className="results-panel">
            <div className="results-table-wrap">
              <div className="panel-heading">
                <div>
                  <span>PACS · Held-out domains</span>
                  <h3>OOD accuracy (%)</h3>
                </div>
                <span className="panel-tag">Source-only adaptation</span>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr><th>Model</th><th>Domain</th><th>FP32</th><th>W4</th><th>Recti-Q</th><th>Recovery</th></tr>
                  </thead>
                  <tbody>
                    {pacsRows.map((row) => (
                      <tr key={`${row[0]}-${row[1]}`}>
                        {row.map((cell, index) => (
                          <td key={cell} className={index === 4 || index === 5 ? "highlight" : ""}>{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="result-figure">
              <img src="/assets/imagenet-c-results.png" alt="ImageNet-C accuracy by corruption and severity for DeiT-S, comparing FP32, W4, and Recti-Q" />
              <p>Recti-Q consistently improves over the W4 baseline across the tested ImageNet-C corruptions.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="deployment section-shell" id="deployment">
        <div className="deployment-card">
          <div className="section-label">Built for deployment</div>
          <h2>Robustness patches small enough to send over the air.</h2>
          <p>
            Once a quantized base model is on-device, only the adapter needs to
            move. That makes Recti-Q practical for low-bandwidth updates across
            robotic fleets operating in changing environments.
          </p>
          <div className="deployment-flow" aria-label="Deployment flow">
            <span>Source data</span><i>→</i><span>Train adapter</span><i>→</i><span>6 KB–0.34 MB</span><i>→</i><span>Edge fleet</span>
          </div>
        </div>
      </section>

      <section className="citation section-shell">
        <div>
          <div className="section-label">Paper</div>
          <h2>Recti-Q</h2>
          <p>
            Feature-Space Rectification for Out-of-Distribution-Robust Quantized
            Perception in Edge Robotics
          </p>
        </div>
        <div className="citation-actions">
          <a className="button button-primary" href="/recti-q-paper.pdf" target="_blank">Download PDF <ArrowIcon /></a>
          <a className="text-link" href="https://github.com/HamidrezaYaghoubi/Recti-Q" target="_blank" rel="noreferrer">Code &amp; pretrained adapters <ArrowIcon /></a>
        </div>
      </section>

      <footer>
        <a className="brand footer-brand" href="#top"><span className="brand-mark">R/Q</span><span>Recti-Q</span></a>
        <p>Robust, efficient perception for the unpredictable physical world.</p>
        <span>© 2026 University of Maryland</span>
      </footer>
    </main>
  );
}
