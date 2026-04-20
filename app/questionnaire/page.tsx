"use client";

import React, { useState, useCallback, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

export default function Questionnaire() {
  const router = useRouter();
  const [phase, setPhase] = useState<'phq9' | 'gad7'>('phq9');
  const [currentStep, setCurrentStep] = useState(0);
  const [phq9Answers, setPhq9Answers] = useState<number[]>([]);
  const [gad7Answers, setGad7Answers] = useState<number[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [slideDirection, setSlideDirection] = useState<'enter' | 'exit' | 'idle'>('enter');
  const [questionKey, setQuestionKey] = useState(0);

  const phq9Questions = [
    "Over the last 2 weeks, how often have you been bothered by having little interest or pleasure in doing things?",
    "Over the last 2 weeks, how often have you been bothered by feeling down, depressed, or hopeless?",
    "Over the last 2 weeks, how often have you been bothered by trouble falling or staying asleep, or sleeping too much?",
    "Over the last 2 weeks, how often have you been bothered by feeling tired or having little energy?",
    "Over the last 2 weeks, how often have you been bothered by poor appetite or overeating?",
    "Over the last 2 weeks, how often have you been bothered by feeling bad about yourself, or that you are a failure?",
    "Over the last 2 weeks, how often have you been bothered by trouble concentrating on things like reading the newspaper?",
    "Over the last 2 weeks, how often have you been bothered by moving or speaking so slowly that other people could have noticed?",
    "Over the last 2 weeks, how often have you been bothered by thoughts that you would be better off dead, or hurting yourself?"
  ];

  const gad7Questions = [
    "Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or on edge?",
    "Over the last 2 weeks, how often have you been bothered by not being able to stop or control worrying?",
    "Over the last 2 weeks, how often have you been bothered by worrying too much about different things?",
    "Over the last 2 weeks, how often have you been bothered by trouble relaxing?",
    "Over the last 2 weeks, how often have you been bothered by being so restless that it's hard to sit still?",
    "Over the last 2 weeks, how often have you been bothered by becoming easily annoyed or irritable?",
    "Over the last 2 weeks, how often have you been bothered by feeling afraid, as if something awful might happen?"
  ];

  const currentQuestions = phase === 'phq9' ? phq9Questions : gad7Questions;

  const questionnaireMeta = {
    phq9: {
      name: "PHQ-9",
      fullName: "Patient Health Questionnaire",
      description: "Depression Screening",
      phaseLabel: "Phase 1: Depression Screening",
      icon: "psychology",
      step: 1,
    },
    gad7: {
      name: "GAD-7",
      fullName: "Generalized Anxiety Disorder Scale",
      description: "Anxiety Screening",
      phaseLabel: "Phase 2: Anxiety Screening",
      icon: "neurology",
      step: 2,
    }
  };

  const meta = questionnaireMeta[phase];

  // Overall progress across both questionnaires
  const totalQuestions = phq9Questions.length + gad7Questions.length;
  const answeredSoFar = phase === 'phq9'
    ? currentStep
    : phq9Questions.length + currentStep;
  const overallProgress = ((answeredSoFar) / totalQuestions) * 100;

  // Local progress within current questionnaire
  const localProgress = ((currentStep) / currentQuestions.length) * 100;

  // Trigger enter animation on mount and step changes
  useEffect(() => {
    setSlideDirection('enter');
  }, [questionKey]);

  const animateTransition = useCallback((callback: () => void) => {
    setSlideDirection('exit');
    setTimeout(() => {
      callback();
      setQuestionKey(prev => prev + 1);
    }, 350);
  }, []);

  const answerOptions = [
    { score: 0, label: "Not at all", sublabel: "0 days" },
    { score: 1, label: "Several days", sublabel: "1–6 days" },
    { score: 2, label: "More than half the days", sublabel: "7–11 days" },
    { score: 3, label: "Nearly every day", sublabel: "12–14 days" },
  ];

  const handleAnswer = async (score: number) => {
    if (isSubmitting || isTransitioning) return;

    if (phase === 'phq9') {
      const newAnswers = [...phq9Answers, score];
      setPhq9Answers(newAnswers);

      if (newAnswers.length === phq9Questions.length) {
        // PHQ-9 complete — submit and transition to GAD-7
        setIsTransitioning(true);

        const payload: Record<string, number> = {};
        newAnswers.forEach((ans, i) => payload[`q${i + 1}`] = ans);

        try {
          const res = await fetch("http://localhost:8000/questionnaire/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          sessionStorage.setItem("phq9_score", data.total_score.toString());

          // Transition to GAD-7 after a brief pause
          setTimeout(() => {
            setPhase('gad7');
            setCurrentStep(0);
            setQuestionKey(prev => prev + 1);
            setIsTransitioning(false);
          }, 1800);
        } catch (err) {
          console.error("Failed to submit PHQ-9", err);
          setIsTransitioning(false);
        }
      } else {
        animateTransition(() => {
          setCurrentStep(prev => prev + 1);
        });
      }
    } else {
      // GAD-7 phase
      const newAnswers = [...gad7Answers, score];
      setGad7Answers(newAnswers);

      if (newAnswers.length === gad7Questions.length) {
        // GAD-7 complete — compute score and go to live session
        setIsSubmitting(true);
        const gad7Score = newAnswers.reduce((a, b) => a + b, 0);
        sessionStorage.setItem("gad7_score", gad7Score.toString());

        // Brief delay then navigate
        setTimeout(() => {
          router.push("/live");
        }, 800);
      } else {
        animateTransition(() => {
          setCurrentStep(prev => prev + 1);
        });
      }
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      animateTransition(() => {
        setCurrentStep(prev => prev - 1);
        if (phase === 'phq9') {
          setPhq9Answers(prev => prev.slice(0, -1));
        } else {
          setGad7Answers(prev => prev.slice(0, -1));
        }
      });
    } else if (phase === 'gad7' && currentStep === 0) {
      // Go back to last PHQ-9 question
      animateTransition(() => {
        setPhase('phq9');
        setCurrentStep(phq9Questions.length - 1);
        setPhq9Answers(prev => prev.slice(0, -1));
      });
    }
  };

  return (
    <>
      {/* ═══════════════════════════════════════════════════════════ */}
      {/*  IMMERSIVE SOFT-FOCUS BACKGROUND                          */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <div className="questionnaire-bg" aria-hidden="true">
        <div className="questionnaire-bg__orb questionnaire-bg__orb--1" />
        <div className="questionnaire-bg__orb questionnaire-bg__orb--2" />
        <div className="questionnaire-bg__orb questionnaire-bg__orb--3" />
        <div className="questionnaire-bg__orb questionnaire-bg__orb--4" />
        <div className="questionnaire-bg__grain" />
      </div>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/*  MINIMAL HEADER                                           */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <header className="fixed top-0 w-full z-50 flex justify-between items-center px-8 py-5">
        <Link href="/" className="font-headline italic text-primary/70 text-xl tracking-tight hover:text-primary transition-colors duration-400">
          Lumina
        </Link>
        <Link href="/">
          <span className="material-symbols-outlined text-on-surface-variant/30 cursor-pointer hover:text-on-surface-variant/60 p-2 rounded-full transition-colors duration-400 text-xl">close</span>
        </Link>
      </header>

      {/* ═══════════════════════════════════════════════════════════ */}
      {/*  PHASE TRANSITION OVERLAY                                 */}
      {/* ═══════════════════════════════════════════════════════════ */}
      {isTransitioning && (
        <div className="questionnaire-transition-overlay">
          <div className="questionnaire-transition-content">
            <div className="questionnaire-transition-icon">
              <span className="material-symbols-outlined text-primary text-3xl" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
            </div>
            <h2 className="font-headline italic text-3xl text-primary mb-4">Phase 1 Complete</h2>
            <p className="font-label text-xs tracking-[0.3em] uppercase text-on-surface-variant/40">
              Preparing anxiety screening…
            </p>
            <div className="questionnaire-transition-progress">
              <div className="questionnaire-transition-progress-fill" />
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════ */}
      {/*  MAIN STAGE                                               */}
      {/* ═══════════════════════════════════════════════════════════ */}
      <main className="flex-grow flex items-center justify-center px-4 sm:px-6 min-h-screen relative z-10">
        <div className="questionnaire-card">

          {/* ─── Progress Bar (top of card) ─── */}
          <div className="questionnaire-card__progress-track">
            <div
              className="questionnaire-card__progress-fill"
              style={{ width: `${overallProgress}%` }}
            />
          </div>

          {/* ─── Phase Indicator ─── */}
          <div className="questionnaire-card__phase">
            <span
              className={`questionnaire-card__phase-dot ${phase === 'phq9' ? 'questionnaire-card__phase-dot--active' : ''}`}
            />
            <span className="questionnaire-card__phase-label">
              {meta.phaseLabel}
            </span>
            <span
              className={`questionnaire-card__phase-dot ${phase === 'gad7' ? 'questionnaire-card__phase-dot--active' : ''}`}
            />
          </div>

          {/* ─── Question Counter ─── */}
          <p className="questionnaire-card__counter">
            Question {currentStep + 1} of {currentQuestions.length}
          </p>

          {/* ─── The Question (Hero Typography) ─── */}
          <div
            key={questionKey}
            className={`questionnaire-card__question-wrapper questionnaire-slide-${slideDirection}`}
          >
            <h1 className="questionnaire-card__question">
              {currentQuestions[currentStep]}
            </h1>
          </div>

          {/* ─── Selection Chips ─── */}
          <div
            key={`chips-${questionKey}`}
            className={`questionnaire-card__chips questionnaire-slide-${slideDirection}`}
            style={{ animationDelay: '0.06s' }}
          >
            {answerOptions.map((option, i) => (
              <button
                key={option.score}
                id={`answer-chip-${option.score}`}
                onClick={() => handleAnswer(option.score)}
                disabled={isSubmitting || isTransitioning}
                className="questionnaire-chip"
                style={{ animationDelay: `${0.08 + i * 0.05}s` }}
              >
                <span className="questionnaire-chip__label">{option.label}</span>
                <span className="questionnaire-chip__sublabel">{option.sublabel}</span>
              </button>
            ))}
          </div>

          {/* ─── Saving Indicator ─── */}
          {isSubmitting && (
            <div className="questionnaire-card__saving">
              <span className="material-symbols-outlined text-primary/60 text-lg animate-pulse mr-2">hourglass_top</span>
              <span className="font-headline italic text-primary/70">Saving your journey…</span>
            </div>
          )}

          {/* ─── Navigation ─── */}
          <div className="questionnaire-card__nav">
            <button
              onClick={handlePrevious}
              className={`questionnaire-nav-btn ${currentStep === 0 && phase === 'phq9' ? 'questionnaire-nav-btn--hidden' : ''}`}
            >
              <span className="material-symbols-outlined text-base">arrow_back</span>
              <span>Previous</span>
            </button>
            <div className="questionnaire-card__step-dots">
              {currentQuestions.map((_, i) => (
                <div
                  key={i}
                  className={`questionnaire-step-dot ${i === currentStep ? 'questionnaire-step-dot--current' : ''} ${i < currentStep ? 'questionnaire-step-dot--done' : ''}`}
                />
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* ─── Footer Watermark ─── */}
      <div className="fixed bottom-3 left-1/2 -translate-x-1/2 opacity-20 pointer-events-none z-10">
        <span className="font-label text-[9px] uppercase tracking-[0.25em] text-on-surface-variant/40">© Lumina Sanctuary</span>
      </div>
    </>
  );
}