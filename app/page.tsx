"use client";

import React, { useEffect, useRef } from 'react';
import Link from 'next/link';

export default function Home() {
  const pageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const el = entry.target as HTMLElement;
            const delay = el.dataset.delay || '0';
            el.style.transitionDelay = `${delay}s`;
            requestAnimationFrame(() => el.classList.add('revealed'));
          }
        });
      },
      { threshold: 0.08, rootMargin: '0px 0px -50px 0px' }
    );

    const elements = pageRef.current?.querySelectorAll('.reveal-on-scroll');
    elements?.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  return (
    <div ref={pageRef} className="overflow-x-hidden">

      {/* ═══════════════════════════════════════════════════════════ */}
      {/*  NAVIGATION                                                */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <nav className="fixed top-0 w-full z-50 bg-surface/60 backdrop-blur-2xl">
        <div className="flex justify-between items-center px-6 md:px-12 lg:px-16 py-5 w-full max-w-screen-2xl mx-auto">
          <Link href="/" className="text-2xl font-headline italic text-primary tracking-tight">Lumina</Link>
          <div className="hidden md:flex gap-10 items-center">
            <Link className="text-primary font-headline italic text-sm tracking-tight border-b border-primary-container/50 pb-0.5" href="#">Sanctuary</Link>
            <Link className="text-secondary/70 hover:text-primary font-headline italic text-sm tracking-tight transition-colors duration-300" href="/report">Progress</Link>
            <Link className="text-secondary/70 hover:text-primary font-headline italic text-sm tracking-tight transition-colors duration-300" href="/live">Breathing</Link>
            <Link className="text-secondary/70 hover:text-primary font-headline italic text-sm tracking-tight transition-colors duration-300" href="/questionnaire">Journal</Link>
          </div>
          <button className="material-symbols-outlined text-primary/60 hover:text-primary transition-colors p-2">account_circle</button>
        </div>
      </nav>

      <main>

        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  HERO — Split-screen layered layout                       */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="relative min-h-screen flex items-center overflow-hidden">
          {/* Atmospheric Background Image — right side */}
          <div className="absolute right-0 top-0 w-[65%] h-full hidden md:block">
            <img
              src="/hero-botanical.webp"
              className="w-full h-full object-cover"
              alt=""
              loading="eager"
            />
          </div>
          {/* Gradient overlays for layered blending */}
          <div className="absolute inset-0 bg-gradient-to-r from-surface via-surface/[0.92] to-surface/20 hidden md:block" />
          <div className="absolute inset-0 bg-gradient-to-b from-surface/80 via-transparent to-surface/60 md:from-transparent md:via-transparent md:to-surface/40" />
          {/* Mobile ambient glow (replaces image on small screens) */}
          <div className="md:hidden absolute inset-0">
            <div className="absolute bottom-1/4 right-0 w-[350px] h-[350px] bg-primary-container/15 rounded-full blur-[120px]" />
            <div className="absolute top-1/3 left-0 w-[250px] h-[250px] bg-secondary/5 rounded-full blur-[100px]" />
          </div>

          {/* Content */}
          <div className="relative z-10 max-w-screen-2xl mx-auto px-6 md:px-12 lg:px-24 w-full pt-32 md:pt-0">
            <div className="max-w-2xl">
              <p className="reveal-on-scroll font-label text-[11px] tracking-[0.35em] uppercase text-secondary/60 mb-8">
                Multimodal Psychological Screening
              </p>
              <h1
                className="reveal-on-scroll font-headline text-[clamp(2.75rem,6.5vw,5.5rem)] italic leading-[1.06] tracking-tight text-primary mb-10"
                data-delay="0.12"
              >
                Your mind deserves more than a questionnaire.
              </h1>
              <p
                className="reveal-on-scroll font-body text-lg md:text-xl text-on-surface-variant/60 max-w-lg leading-relaxed tracking-wide mb-14"
                data-delay="0.24"
              >
                Lumina is a private, AI-guided wellness screening that listens to your voice, reads your expressions, and sees what words alone cannot capture.
              </p>
              <div className="reveal-on-scroll" data-delay="0.36">
                <Link href="/questionnaire">
                  <button className="cta-glass group px-10 py-[1.15rem] rounded-2xl text-[15px] font-medium tracking-wide flex items-center gap-3">
                    Begin Your Screening
                    <span className="material-symbols-outlined text-lg opacity-70 group-hover:translate-x-1 group-hover:opacity-100 transition-all duration-300">arrow_forward</span>
                  </button>
                </Link>
                <p className="mt-6 text-[10px] font-label tracking-[0.3em] uppercase text-secondary/30">
                  15 minutes · Free · Private
                </p>
              </div>
            </div>
          </div>

          {/* Scroll indicator */}
          <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 animate-pulse">
            <span className="text-[9px] font-label tracking-[0.4em] uppercase text-secondary/25">Scroll</span>
            <div className="w-px h-10 bg-gradient-to-b from-primary/15 to-transparent" />
          </div>
        </section>


        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  EDITORIAL STATS STRIP                                    */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="py-16 md:py-20 px-6 md:px-16">
          <div className="reveal-on-scroll max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-0">
            <div className="text-center md:border-r md:border-outline-variant/10 px-4 md:px-8">
              <div className="text-4xl md:text-5xl font-headline italic text-primary mb-2">1 in 4</div>
              <div className="text-xs text-on-surface-variant/40 font-body leading-relaxed">people affected by mental disorders in their lifetime</div>
            </div>
            <div className="text-center md:border-r md:border-outline-variant/10 px-4 md:px-8">
              <div className="text-4xl md:text-5xl font-headline italic text-primary mb-2">280M+</div>
              <div className="text-xs text-on-surface-variant/40 font-body leading-relaxed">people worldwide live with depression</div>
            </div>
            <div className="text-center md:border-r md:border-outline-variant/10 px-4 md:px-8">
              <div className="text-4xl md:text-5xl font-headline italic text-primary mb-2">50%</div>
              <div className="text-xs text-on-surface-variant/40 font-body leading-relaxed">of conditions begin by age 14</div>
            </div>
            <div className="text-center px-4 md:px-8">
              <div className="text-4xl md:text-5xl font-headline italic text-primary mb-2">&lt;1 in 3</div>
              <div className="text-xs text-on-surface-variant/40 font-body leading-relaxed">seek professional help</div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  THE INVISIBLE GAP IN WELLNESS                            */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="py-28 md:py-40 px-6 md:px-16">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-20 md:mb-24">
              <span className="reveal-on-scroll inline-block font-label text-[11px] tracking-[0.35em] uppercase text-secondary/50 mb-8">
                The Problem
              </span>
              <h2
                className="reveal-on-scroll font-headline text-4xl md:text-5xl lg:text-[3.5rem] italic text-primary leading-tight mb-8"
                data-delay="0.1"
              >
                The Invisible Gap in Wellness
              </h2>
              <p
                className="reveal-on-scroll font-body text-lg text-on-surface-variant/50 max-w-2xl mx-auto leading-relaxed"
                data-delay="0.2"
              >
                Traditional mental health assessments rely on subjective, single-modality evaluations.
                They capture what you say — but miss how you say it, what your face reveals,
                and the patterns hidden between the lines.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-10">
              <div className="reveal-on-scroll text-center p-8 md:p-10 rounded-3xl bg-surface-container-lowest/60 shadow-[0_16px_48px_-12px_rgba(46,41,37,0.04)]" data-delay="0.1">
                <div className="w-16 h-16 mx-auto mb-7 rounded-2xl bg-surface-container flex items-center justify-center">
                  <span className="material-symbols-outlined text-2xl text-primary/40">psychology_alt</span>
                </div>
                <h4 className="font-headline text-lg italic text-on-surface mb-3">Subjective Evaluation</h4>
                <p className="text-sm text-on-surface-variant/40 leading-relaxed font-body">
                  Manual assessments vary between clinicians, introducing inconsistency and bias into critical diagnoses.
                </p>
              </div>
              <div className="reveal-on-scroll text-center p-8 md:p-10 rounded-3xl bg-surface-container-lowest/60 shadow-[0_16px_48px_-12px_rgba(46,41,37,0.04)]" data-delay="0.2">
                <div className="w-16 h-16 mx-auto mb-7 rounded-2xl bg-surface-container flex items-center justify-center">
                  <span className="material-symbols-outlined text-2xl text-primary/40">view_in_ar</span>
                </div>
                <h4 className="font-headline text-lg italic text-on-surface mb-3">Single Modality</h4>
                <p className="text-sm text-on-surface-variant/40 leading-relaxed font-body">
                  Existing tools analyze only one signal — missing the full multi-dimensional picture of emotional state.
                </p>
              </div>
              <div className="reveal-on-scroll text-center p-8 md:p-10 rounded-3xl bg-surface-container-lowest/60 shadow-[0_16px_48px_-12px_rgba(46,41,37,0.04)]" data-delay="0.3">
                <div className="w-16 h-16 mx-auto mb-7 rounded-2xl bg-surface-container flex items-center justify-center">
                  <span className="material-symbols-outlined text-2xl text-primary/40">scale</span>
                </div>
                <h4 className="font-headline text-lg italic text-on-surface mb-3">Not Scalable</h4>
                <p className="text-sm text-on-surface-variant/40 leading-relaxed font-body">
                  One-on-one clinical sessions are expensive, time-consuming, and inaccessible to millions.
                </p>
              </div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  THE SOLUTION — Multimodal Correlation                    */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="py-28 md:py-40 px-6 md:px-16 relative overflow-hidden">
          {/* Subtle tonal background shift */}
          <div className="absolute inset-0 bg-gradient-to-b from-surface-container-low/30 via-surface-container-low/50 to-surface" />

          <div className="relative z-10 max-w-6xl mx-auto">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 lg:gap-24 items-center">
              {/* Copy */}
              <div>
                <span className="reveal-on-scroll inline-block font-label text-[11px] tracking-[0.35em] uppercase text-secondary/50 mb-8">
                  The Solution
                </span>
                <h2
                  className="reveal-on-scroll font-headline text-4xl md:text-5xl italic text-primary leading-tight mb-8"
                  data-delay="0.1"
                >
                  Multimodal<br />Correlation
                </h2>
                <p
                  className="reveal-on-scroll font-body text-lg text-on-surface-variant/55 leading-relaxed mb-10"
                  data-delay="0.2"
                >
                  Lumina doesn&apos;t just ask how you feel — it observes the full spectrum.
                  By correlating self-reported responses with real-time vocal biomarkers
                  and facial micro-expressions, we build a unified, multi-dimensional
                  understanding of emotional state.
                </p>
                <div className="reveal-on-scroll space-y-5" data-delay="0.3">
                  <div className="flex items-center gap-4">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/30 flex-shrink-0" />
                    <span className="font-body text-sm text-on-surface-variant/55 tracking-wide">Cross-modal consistency scoring</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/30 flex-shrink-0" />
                    <span className="font-body text-sm text-on-surface-variant/55 tracking-wide">Emotional reactivity analysis</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-1.5 h-1.5 rounded-full bg-primary/30 flex-shrink-0" />
                    <span className="font-body text-sm text-on-surface-variant/55 tracking-wide">Weighted multimodal fusion</span>
                  </div>
                </div>
              </div>

              {/* Visual — Three channels converging into unified assessment */}
              <div className="reveal-on-scroll" data-delay="0.15">
                <div className="relative bg-surface-container-lowest rounded-[2rem] p-10 md:p-12 shadow-[0_24px_64px_-12px_rgba(113,91,58,0.07)]">
                  {/* Three input channels */}
                  <div className="flex flex-col gap-5 mb-10">
                    <div className="flex items-center gap-4">
                      <div className="w-11 h-11 rounded-xl bg-primary-container/15 flex items-center justify-center flex-shrink-0">
                        <span className="material-symbols-outlined text-lg text-primary/70">quiz</span>
                      </div>
                      <span className="font-label text-[10px] tracking-[0.2em] uppercase text-on-surface-variant/35 flex-shrink-0 w-24">Questionnaire</span>
                      <div className="flex-1 h-px bg-gradient-to-r from-primary-container/30 to-primary/10 rounded-full" />
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="w-11 h-11 rounded-xl bg-primary-container/15 flex items-center justify-center flex-shrink-0">
                        <span className="material-symbols-outlined text-lg text-primary/70">mic</span>
                      </div>
                      <span className="font-label text-[10px] tracking-[0.2em] uppercase text-on-surface-variant/35 flex-shrink-0 w-24">Voice</span>
                      <div className="flex-1 h-px bg-gradient-to-r from-primary-container/30 to-primary/10 rounded-full" />
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="w-11 h-11 rounded-xl bg-primary-container/15 flex items-center justify-center flex-shrink-0">
                        <span className="material-symbols-outlined text-lg text-primary/70">face</span>
                      </div>
                      <span className="font-label text-[10px] tracking-[0.2em] uppercase text-on-surface-variant/35 flex-shrink-0 w-24">Facial</span>
                      <div className="flex-1 h-px bg-gradient-to-r from-primary-container/30 to-primary/10 rounded-full" />
                    </div>
                  </div>

                  {/* Thin separator */}
                  <div className="w-full h-px bg-outline-variant/10 mb-8" />

                  {/* Convergence point */}
                  <div className="flex items-center justify-center">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-primary/[0.06] to-primary-container/15 flex items-center justify-center shadow-[0_8px_32px_rgba(113,91,58,0.08)]">
                      <span className="material-symbols-outlined text-3xl text-primary/50" style={{ fontVariationSettings: "'FILL' 1" }}>hub</span>
                    </div>
                  </div>
                  <p className="text-center mt-5 font-label text-[10px] tracking-[0.25em] uppercase text-on-surface-variant/30">
                    Unified Assessment
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  MODALITY SHOWCASE — Asymmetric Cards with Hover Reveals  */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="py-28 md:py-40 px-6 md:px-16">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-20">
              <span className="reveal-on-scroll font-label text-[11px] tracking-[0.35em] uppercase text-secondary/50 mb-6 block">
                Capabilities
              </span>
              <h2
                className="reveal-on-scroll font-headline text-4xl md:text-5xl italic text-primary"
                data-delay="0.1"
              >
                Three Lenses, One Truth
              </h2>
            </div>

            {/* Asymmetric card grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8 items-start">
              {/* Card 1 — Questionnaire */}
              <div className="reveal-on-scroll md:mt-0" data-delay="0.1">
                <div className="group relative bg-surface-container-lowest rounded-3xl p-9 md:p-10 shadow-[0_20px_60px_-10px_rgba(46,41,37,0.05)] overflow-hidden transition-all duration-500 hover:shadow-[0_28px_70px_-8px_rgba(113,91,58,0.1)] hover:-translate-y-1.5">
                  <div className="relative z-10">
                    <div className="w-14 h-14 rounded-2xl bg-primary-container/10 flex items-center justify-center mb-8 transition-colors duration-300 group-hover:bg-primary-container/20">
                      <span className="material-symbols-outlined text-2xl text-primary/70" style={{ fontVariationSettings: "'FILL' 1" }}>quiz</span>
                    </div>
                    <h3 className="font-headline text-[1.4rem] italic text-on-surface mb-4">Clinical Questionnaires</h3>
                    <p className="font-body text-sm text-on-surface-variant/50 leading-relaxed mb-8">
                      Standardized PHQ-9 and GAD-7 screenings, reimagined as a meditative, judgment-free digital experience.
                    </p>
                  </div>
                  {/* Tech reveal on hover */}
                  <div className="absolute inset-x-0 bottom-0 p-5 pt-8 bg-gradient-to-t from-surface-container-lowest via-surface-container-lowest/95 to-transparent translate-y-full group-hover:translate-y-0 transition-transform duration-500 ease-out">
                    <div className="flex items-center gap-2 text-[10px] font-label tracking-[0.15em] text-primary/50 uppercase">
                      <span className="material-symbols-outlined text-xs">code</span>
                      Clinically validated · 0–27 severity mapping · PHQ-9 + GAD-7
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 2 — Voice (offset for asymmetry) */}
              <div className="reveal-on-scroll md:mt-14" data-delay="0.2">
                <div className="group relative bg-surface-container-lowest rounded-3xl p-9 md:p-10 shadow-[0_20px_60px_-10px_rgba(46,41,37,0.05)] overflow-hidden transition-all duration-500 hover:shadow-[0_28px_70px_-8px_rgba(113,91,58,0.1)] hover:-translate-y-1.5">
                  <div className="relative z-10">
                    <div className="w-14 h-14 rounded-2xl bg-primary-container/10 flex items-center justify-center mb-8 transition-colors duration-300 group-hover:bg-primary-container/20">
                      <span className="material-symbols-outlined text-2xl text-primary/70" style={{ fontVariationSettings: "'FILL' 1" }}>settings_voice</span>
                    </div>
                    <h3 className="font-headline text-[1.4rem] italic text-on-surface mb-4">Voice Behavioral Analysis</h3>
                    <p className="font-body text-sm text-on-surface-variant/50 leading-relaxed mb-8">
                      AI detects acoustic biomarkers in your speech — energy, pitch stability, and emotional tone that correlate with internal states.
                    </p>
                  </div>
                  <div className="absolute inset-x-0 bottom-0 p-5 pt-8 bg-gradient-to-t from-surface-container-lowest via-surface-container-lowest/95 to-transparent translate-y-full group-hover:translate-y-0 transition-transform duration-500 ease-out">
                    <div className="flex items-center gap-2 text-[10px] font-label tracking-[0.15em] text-primary/50 uppercase">
                      <span className="material-symbols-outlined text-xs">code</span>
                      CNN + Dense hybrid · MFCC + Mel Spectrogram · Temporal smoothing
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 3 — Facial (smaller offset) */}
              <div className="reveal-on-scroll md:mt-5" data-delay="0.3">
                <div className="group relative bg-surface-container-lowest rounded-3xl p-9 md:p-10 shadow-[0_20px_60px_-10px_rgba(46,41,37,0.05)] overflow-hidden transition-all duration-500 hover:shadow-[0_28px_70px_-8px_rgba(113,91,58,0.1)] hover:-translate-y-1.5">
                  <div className="relative z-10">
                    <div className="w-14 h-14 rounded-2xl bg-primary-container/10 flex items-center justify-center mb-8 transition-colors duration-300 group-hover:bg-primary-container/20">
                      <span className="material-symbols-outlined text-2xl text-primary/70" style={{ fontVariationSettings: "'FILL' 1" }}>face</span>
                    </div>
                    <h3 className="font-headline text-[1.4rem] italic text-on-surface mb-4">Facial Emotion Detection</h3>
                    <p className="font-body text-sm text-on-surface-variant/50 leading-relaxed mb-8">
                      Subtle micro-expression tracking maps the landscape of your inner world through real-time facial analysis.
                    </p>
                  </div>
                  <div className="absolute inset-x-0 bottom-0 p-5 pt-8 bg-gradient-to-t from-surface-container-lowest via-surface-container-lowest/95 to-transparent translate-y-full group-hover:translate-y-0 transition-transform duration-500 ease-out">
                    <div className="flex items-center gap-2 text-[10px] font-label tracking-[0.15em] text-primary/50 uppercase">
                      <span className="material-symbols-outlined text-xs">code</span>
                      DeepFace engine · 7-emotion classification · Confidence scoring
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  THE JOURNEY                                              */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="py-28 md:py-40 px-6 md:px-16 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-surface via-surface-container-low/30 to-surface" />

          <div className="relative z-10 max-w-5xl mx-auto">
            <h2 className="reveal-on-scroll font-headline text-4xl md:text-5xl italic text-primary text-center mb-6">
              Your Journey to Clarity
            </h2>
            <p className="reveal-on-scroll font-body text-on-surface-variant/40 text-center max-w-xl mx-auto mb-20 md:mb-24" data-delay="0.1">
              A calm, three-phase experience designed around your comfort.
            </p>

            <div className="space-y-16 md:space-y-0 md:grid md:grid-cols-3 md:gap-12 lg:gap-16 relative">
              {/* Connector line */}
              <div className="hidden md:block absolute top-14 left-[16.67%] right-[16.67%] h-px bg-outline-variant/15" />

              <div className="reveal-on-scroll text-center" data-delay="0.1">
                <div className="w-[6.5rem] h-[6.5rem] mx-auto rounded-full bg-surface-container-lowest shadow-[0_12px_40px_-8px_rgba(46,41,37,0.06)] flex items-center justify-center mb-8 relative z-10">
                  <span className="font-headline text-3xl italic text-primary">1</span>
                </div>
                <h4 className="font-headline text-xl italic text-on-surface mb-3">Reflect</h4>
                <p className="font-body text-sm text-on-surface-variant/45 leading-relaxed max-w-xs mx-auto">
                  Clinical self-assessment through PHQ-9 and GAD-7, reimagined as a calm, meditative experience.
                </p>
              </div>

              <div className="reveal-on-scroll text-center" data-delay="0.2">
                <div className="w-[6.5rem] h-[6.5rem] mx-auto rounded-full bg-surface-container-lowest shadow-[0_12px_40px_-8px_rgba(46,41,37,0.06)] flex items-center justify-center mb-8 relative z-10">
                  <span className="font-headline text-3xl italic text-primary">2</span>
                </div>
                <h4 className="font-headline text-xl italic text-on-surface mb-3">Discover</h4>
                <p className="font-body text-sm text-on-surface-variant/45 leading-relaxed max-w-xs mx-auto">
                  Engage in a guided AI session. Your voice and expressions are analyzed in real-time for emotional insight.
                </p>
              </div>

              <div className="reveal-on-scroll text-center" data-delay="0.3">
                <div className="w-[6.5rem] h-[6.5rem] mx-auto rounded-full bg-surface-container-lowest shadow-[0_12px_40px_-8px_rgba(46,41,37,0.06)] flex items-center justify-center mb-8 relative z-10">
                  <span className="font-headline text-3xl italic text-primary">3</span>
                </div>
                <h4 className="font-headline text-xl italic text-on-surface mb-3">Understand</h4>
                <p className="font-body text-sm text-on-surface-variant/45 leading-relaxed max-w-xs mx-auto">
                  Receive an editorial report — a unified, AI-generated narrative of your emotional landscape.
                </p>
              </div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  TRUST & PRIVACY                                          */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="py-28 md:py-40 px-6 md:px-16 relative overflow-hidden">
          {/* Giant shield watermark */}
          <div className="absolute right-0 top-1/2 -translate-y-1/2 opacity-[0.02] pointer-events-none">
            <span className="material-symbols-outlined text-[28rem] text-primary" style={{ fontVariationSettings: "'FILL' 1" }}>shield</span>
          </div>

          <div className="max-w-6xl mx-auto relative z-10">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 lg:gap-24 items-center">
              <div>
                <span className="reveal-on-scroll inline-block font-label text-[11px] tracking-[0.35em] uppercase text-secondary/50 mb-8">
                  Trust &amp; Privacy
                </span>
                <h2
                  className="reveal-on-scroll font-headline text-4xl md:text-5xl italic text-primary leading-tight mb-12"
                  data-delay="0.1"
                >
                  Your privacy is sacred.
                </h2>
                <div className="space-y-9">
                  <div className="reveal-on-scroll flex gap-5" data-delay="0.15">
                    <div className="w-11 h-11 rounded-xl bg-surface-container flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="material-symbols-outlined text-lg text-primary/50">lock</span>
                    </div>
                    <div>
                      <h4 className="font-body font-semibold text-sm text-on-surface mb-1">End-to-end encryption</h4>
                      <p className="text-sm text-on-surface-variant/40 leading-relaxed font-body">
                        Your session data is encrypted in transit and at rest. Wellness data stays yours alone.
                      </p>
                    </div>
                  </div>
                  <div className="reveal-on-scroll flex gap-5" data-delay="0.25">
                    <div className="w-11 h-11 rounded-xl bg-surface-container flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="material-symbols-outlined text-lg text-primary/50">cloud_off</span>
                    </div>
                    <div>
                      <h4 className="font-body font-semibold text-sm text-on-surface mb-1">Local processing</h4>
                      <p className="text-sm text-on-surface-variant/40 leading-relaxed font-body">
                        Audio and facial analysis runs locally on-device. Biometric data never leaves your machine.
                      </p>
                    </div>
                  </div>
                  <div className="reveal-on-scroll flex gap-5" data-delay="0.35">
                    <div className="w-11 h-11 rounded-xl bg-surface-container flex items-center justify-center flex-shrink-0 mt-0.5">
                      <span className="material-symbols-outlined text-lg text-primary/50">verified_user</span>
                    </div>
                    <div>
                      <h4 className="font-body font-semibold text-sm text-on-surface mb-1">Ethical AI guidelines</h4>
                      <p className="text-sm text-on-surface-variant/40 leading-relaxed font-body">
                        Built with strict ethical constraints. No data selling, no tracking, no profiling.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Abstract visual — nested circles */}
              <div className="reveal-on-scroll hidden lg:flex items-center justify-center" data-delay="0.2">
                <div className="relative">
                  <div className="w-80 h-80 rounded-full bg-gradient-to-br from-surface-container-low to-surface-container/50 flex items-center justify-center">
                    <div className="w-52 h-52 rounded-full bg-surface-container-lowest shadow-[0_16px_56px_-12px_rgba(113,91,58,0.08)] flex items-center justify-center">
                      <span className="material-symbols-outlined text-6xl text-primary/20" style={{ fontVariationSettings: "'FILL' 1" }}>shield</span>
                    </div>
                  </div>
                  {/* Floating accent dots */}
                  <div className="absolute top-6 right-8 w-3 h-3 rounded-full bg-primary-container/30 animate-pulse" />
                  <div className="absolute bottom-12 left-2 w-2 h-2 rounded-full bg-tertiary/20" />
                  <div className="absolute top-1/2 -right-1 w-4 h-4 rounded-full bg-secondary/8" />
                </div>
              </div>
            </div>
          </div>
        </section>


        {/* ═══════════════════════════════════════════════════════════ */}
        {/*  CLOSING CTA                                              */}
        {/* ═══════════════════════════════════════════════════════════ */}
        <section className="py-20 md:py-32 px-6">
          <div className="max-w-4xl mx-auto">
            <div className="reveal-on-scroll relative bg-surface-container-lowest rounded-[2.5rem] p-12 md:p-20 lg:p-24 shadow-[0_24px_80px_-16px_rgba(113,91,58,0.06)] overflow-hidden text-center">
              {/* Ambient glows */}
              <div className="absolute top-0 right-0 w-80 h-80 bg-primary-container/8 rounded-full blur-[100px] -translate-y-1/2 translate-x-1/4 pointer-events-none" />
              <div className="absolute bottom-0 left-0 w-60 h-60 bg-secondary/4 rounded-full blur-[80px] translate-y-1/3 -translate-x-1/4 pointer-events-none" />

              <div className="relative z-10">
                <h2 className="font-headline text-4xl md:text-5xl lg:text-[3.5rem] italic text-primary leading-tight mb-8">
                  Ready to see yourself clearly?
                </h2>
                <p className="font-body text-base md:text-lg text-on-surface-variant/45 max-w-xl mx-auto mb-12 leading-relaxed">
                  No judgment. No diagnosis. Just a deeper, multi-dimensional understanding of your emotional wellness.
                </p>
                <Link href="/questionnaire">
                  <button className="cta-glass px-12 py-[1.15rem] rounded-2xl text-[15px] font-medium tracking-wide">
                    Begin Your Screening
                  </button>
                </Link>
                <p className="text-[10px] font-label tracking-[0.3em] uppercase text-secondary/25 mt-6">
                  15 minutes · Free · Private
                </p>
              </div>
            </div>
          </div>
        </section>

      </main>


      {/* ═══════════════════════════════════════════════════════════ */}
      {/*  FOOTER                                                    */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <footer className="px-6 md:px-16 py-14 mt-8">
        <div className="max-w-screen-2xl mx-auto">
          {/* Thin top accent line */}
          <div className="w-12 h-px bg-outline-variant/15 mb-10" />
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
            <div>
              <div className="font-headline text-lg italic text-primary mb-1.5">Lumina</div>
              <div className="font-body text-[11px] tracking-[0.15em] text-on-surface-variant/25 uppercase">
                © 2024 Lumina Sanctuary · Your private wellness companion
              </div>
            </div>
            <div className="flex gap-8 items-center">
              <a className="font-body text-[11px] tracking-[0.12em] text-on-surface-variant/25 hover:text-primary transition-colors duration-300 uppercase" href="#">Privacy</a>
              <a className="font-body text-[11px] tracking-[0.12em] text-on-surface-variant/25 hover:text-primary transition-colors duration-300 uppercase" href="#">Terms</a>
              <a className="font-body text-[11px] tracking-[0.12em] text-on-surface-variant/25 hover:text-primary transition-colors duration-300 uppercase" href="#">Clinical Disclaimer</a>
            </div>
          </div>
        </div>
      </footer>

    </div>
  );
}