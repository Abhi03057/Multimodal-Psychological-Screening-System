"use client";

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import ReactMarkdown from 'react-markdown';

export default function Report() {
  const router = useRouter();
  const [reportData, setReportData] = useState<any>(null);

  useEffect(() => {
    // Load the generated session report from the browser session
    const savedData = sessionStorage.getItem("lumina_report");
    if (savedData) {
      try {
        setReportData(JSON.parse(savedData));
      } catch(err) {
        console.error("Could not parse report data");
      }
    }
  }, []);

  // Return loading screen if we haven't loaded the data yet
  if (!reportData) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-surface">
        <div className="text-primary italic font-serif text-2xl animate-pulse">
          Retrieving your sanctuary report...
        </div>
      </div>
    );
  }

  // Derive simple UI metrics from backend normalizations
  const stabilityScore = Math.round((1 - (reportData.aggregated?.final_score || 0)) * 100) || 84;
  const isHighRisk = reportData.aggregated?.risk_level === "High" || reportData.phq9_score > 14;

  const handleStartOver = () => {
    sessionStorage.removeItem("phq9_score");
    sessionStorage.removeItem("gad7_score");
    sessionStorage.removeItem("lumina_report");
    router.push("/");
  };

  return (
    <>
      <header className="fixed top-0 w-full z-50 bg-[#fcf9f6]/72 dark:bg-[#2E2925]/72 backdrop-blur-xl shadow-[0_8px_32px_rgba(46,41,37,0.06)]">
        <nav className="flex justify-between items-center px-12 py-6 w-full max-w-screen-2xl mx-auto">
          <div className="flex items-center gap-12">
            <span className="text-2xl font-serif italic text-[#715b3a] dark:text-[#c4a882]">Lumina</span>
            <div className="hidden md:flex gap-8 items-center">
              <Link className="text-[#436275] dark:text-[#e5e2df] hover:text-[#715b3a] font-sans text-sm tracking-wide transition-opacity duration-300" href="/">Sanctuary</Link>
              <Link className="text-[#715b3a] dark:text-[#c4a882] font-semibold border-b-2 border-[#c4a882] pb-1 font-sans text-sm tracking-wide" href="#">Progress</Link>
              <Link className="text-[#436275] dark:text-[#e5e2df] hover:text-[#715b3a] font-sans text-sm tracking-wide transition-opacity duration-300" href="/live">Breathing</Link>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <button className="hover:opacity-80 transition-opacity duration-300 scale-95">
              <span className="material-symbols-outlined text-[#715b3a] text-3xl">account_circle</span>
            </button>
          </div>
        </nav>
      </header>

      <main className="pt-32 pb-20 px-6 max-w-5xl mx-auto">
        <div className="mb-12 text-center">
          <h1 className="font-headline text-5xl md:text-6xl italic tracking-tight text-primary mb-4">Wellness Report</h1>
          <p className="font-body text-secondary opacity-70">A reflective analysis of your recent multimodal session.</p>
        </div>
        
        {/* Core Summary */}
        <section className="mb-10">
          <div className="bg-surface-container-lowest p-10 md:p-14 rounded-lg shadow-[0_8px_32px_rgba(46,41,37,0.06)] flex flex-col md:flex-row items-center justify-between gap-8 border border-outline-variant/10">
            <div className="max-w-xl">
              <div className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6 ${isHighRisk ? 'bg-error-container text-on-error-container' : 'bg-tertiary-container/20 text-tertiary'}`}>
                <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>
                  {isHighRisk ? 'warning' : 'eco'}
                </span>
                <span className="text-xs font-semibold tracking-widest uppercase">
                  {reportData.aggregated?.risk_level || "Unknown"} Risk
                </span>
              </div>
              <h2 className="font-headline text-3xl text-on-surface mb-4">Multimodal Insights</h2>
              <p className="text-on-surface-variant leading-relaxed">
                By combining your self-reported clinical scores with dynamic analysis of your vocal tone and facial expressions, Lumina generated a unified view of your emotional resilience during the session.
              </p>
            </div>
            <div className="relative w-48 h-48 flex items-center justify-center">
              <div className="absolute inset-0 bg-primary/5 rounded-full animate-pulse"></div>
              <div className="text-center">
                <div className="text-5xl font-headline italic text-primary">{stabilityScore}</div>
                <div className="text-xs tracking-widest uppercase opacity-50 font-semibold">Stability Score</div>
              </div>
            </div>
          </div>
        </section>

        {/* Clinical Grid */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
          <div className="bg-surface-container-low p-8 rounded-lg border border-outline-variant/10">
            <div className="flex justify-between items-end mb-6">
              <div>
                <h3 className="font-headline text-2xl text-on-surface">PHQ-9</h3>
                <p className="text-xs text-on-surface-variant opacity-60">Depression Severity</p>
              </div>
              <div className="text-right">
                <span className="text-2xl font-headline text-primary">{reportData.phq9_score}</span>
                <span className="text-sm opacity-40">/27</span>
              </div>
            </div>
            <div className="h-1.5 w-full bg-outline-variant/20 rounded-full mb-4 overflow-hidden">
              <div className="h-full bg-primary-container rounded-full" style={{ width: `${(reportData.phq9_score / 27) * 100}%` }}></div>
            </div>
          </div>
          
          <div className="bg-surface-container-low p-8 rounded-lg border border-outline-variant/10">
            <div className="flex justify-between items-end mb-6">
              <div>
                <h3 className="font-headline text-2xl text-on-surface">GAD-7</h3>
                <p className="text-xs text-on-surface-variant opacity-60">Anxiety Severity</p>
              </div>
              <div className="text-right">
                <span className="text-2xl font-headline text-primary">{reportData.gad7_score}</span>
                <span className="text-sm opacity-40">/21</span>
              </div>
            </div>
            <div className="h-1.5 w-full bg-outline-variant/20 rounded-full mb-4 overflow-hidden">
              <div className="h-full bg-tertiary-container rounded-full" style={{ width: `${(reportData.gad7_score / 21) * 100}%` }}></div>
            </div>
          </div>
        </section>

        {/* LLM Narrative Section */}
        <section className="relative p-10 md:p-16 mb-16 editorial-accent">
          <div className="max-w-3xl">
            <h3 className="font-headline text-sm tracking-widest uppercase opacity-40 mb-6">Lumina AI Insight</h3>
            <div className="prose prose-stone max-w-none text-on-surface-variant leading-loose [&_h1]:font-headline [&_h2]:font-headline [&_h3]:font-headline [&_h1]:italic [&_h2]:italic [&_strong]:text-on-surface">
              <ReactMarkdown>
                {reportData.report || "No personalized LLM report was generated for this session."}
              </ReactMarkdown>
            </div>
            <div className="mt-10 flex items-center gap-4">
              <div className="w-10 h-1 bg-primary-container rounded-full"></div>
              <span className="font-headline italic text-on-surface opacity-60 text-lg">Your AI Care Companion, Lumina</span>
            </div>
          </div>
        </section>

        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <button onClick={() => window.print()} className="bg-surface-container text-on-surface px-10 py-4 rounded-xl font-medium flex items-center justify-center gap-3 hover:opacity-80 shadow border border-outline-variant/20">
            <span className="material-symbols-outlined">file_download</span>
            Download PDF
          </button>
          <button onClick={handleStartOver} className="bg-primary text-on-primary px-10 py-4 rounded-xl font-medium flex items-center justify-center gap-3 hover:opacity-90 shadow-lg">
            <span className="material-symbols-outlined">restart_alt</span>
            Start Over
          </button>
        </div>
      </main>
    </>
  );
}