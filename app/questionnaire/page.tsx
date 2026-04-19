"use client";

import React, { useState } from 'react';
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
  const currentAnswers = phase === 'phq9' ? phq9Answers : gad7Answers;

  const questionnaireMeta = {
    phq9: {
      name: "PHQ-9",
      fullName: "Patient Health Questionnaire",
      description: "Depression Screening",
      icon: "psychology",
      step: 1,
    },
    gad7: {
      name: "GAD-7",
      fullName: "Generalized Anxiety Disorder Scale",
      description: "Anxiety Screening",
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
  const overallProgress = Math.round(((answeredSoFar + 1) / totalQuestions) * 100);

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
            setIsTransitioning(false);
          }, 1200);
        } catch (err) {
          console.error("Failed to submit PHQ-9", err);
          setIsTransitioning(false);
        }
      } else {
        setCurrentStep(prev => prev + 1);
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
        setCurrentStep(prev => prev + 1);
      }
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
      if (phase === 'phq9') {
        setPhq9Answers(prev => prev.slice(0, -1));
      } else {
        setGad7Answers(prev => prev.slice(0, -1));
      }
    } else if (phase === 'gad7' && currentStep === 0) {
      // Go back to last PHQ-9 question
      setPhase('phq9');
      setCurrentStep(phq9Questions.length - 1);
      setPhq9Answers(prev => prev.slice(0, -1));
    }
  };

  const localProgress = Math.round(((currentStep + 1) / currentQuestions.length) * 100);

  return (
    <>
      <header className="fixed top-0 w-full z-50 flex justify-between items-center px-8 py-4 bg-stone-50/72 backdrop-blur-md">
        <div className="font-serif italic text-stone-800 text-2xl tracking-tight">Lumina</div>
        <div className="flex items-center gap-4">
          <Link href="/">
            <span className="material-symbols-outlined text-stone-500 cursor-pointer hover:bg-stone-100/50 p-2 rounded-full transition-colors duration-400">close</span>
          </Link>
        </div>
      </header>
      
      <main className="flex-grow flex flex-col items-center justify-center px-6 pt-24 pb-32 min-h-screen relative z-10">

        {/* Transition overlay */}
        {isTransitioning && (
          <div className="fixed inset-0 z-40 bg-stone-50/90 backdrop-blur-lg flex flex-col items-center justify-center animate-fade-in">
            <div className="w-16 h-16 rounded-full bg-primary-container/20 flex items-center justify-center mb-6 animate-pulse">
              <span className="material-symbols-outlined text-primary text-3xl">check_circle</span>
            </div>
            <h2 className="font-serif italic text-2xl text-primary mb-3">PHQ-9 Complete</h2>
            <p className="text-stone-500 font-label text-sm tracking-wide">Preparing anxiety screening...</p>
          </div>
        )}
        
        {/* Questionnaire Badge */}
        <div className="w-full max-w-2xl mb-6 flex flex-col items-center">
          <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full bg-primary-container/15 border border-primary-container/25 mb-6">
            <span className="material-symbols-outlined text-primary text-lg" style={{ fontVariationSettings: "'FILL' 1" }}>{meta.icon}</span>
            <div className="flex flex-col">
              <span className="font-label text-xs tracking-widest uppercase text-primary font-semibold">
                Questionnaire {meta.step} of 2 — {meta.name}
              </span>
              <span className="font-body text-[11px] text-stone-500">
                {meta.fullName} · {meta.description}
              </span>
            </div>
          </div>
        </div>

        {/* Progress section */}
        <div className="w-full max-w-2xl mb-12 flex flex-col items-center">
          {/* Overall progress (thin) */}
          <div className="w-full flex items-center gap-3 mb-3">
            <span className="font-label text-[10px] tracking-widest uppercase text-stone-400 whitespace-nowrap">Overall</span>
            <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
              <div 
                  className="h-full bg-stone-400/40 rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${overallProgress}%` }}
              ></div>
            </div>
            <span className="font-label text-[10px] text-stone-400">{overallProgress}%</span>
          </div>

          {/* Current questionnaire progress (prominent) */}
          <div className="w-full h-1.5 bg-surface-container-highest rounded-full overflow-hidden mb-4">
            <div 
                className="h-full bg-primary-container rounded-full transition-all duration-700 ease-out shadow-[0_0_8px_rgba(113,91,58,0.2)]"
                style={{ width: `${localProgress}%` }}
            ></div>
          </div>
          <span className="font-label text-sm tracking-widest uppercase text-stone-400 font-medium">
            {meta.name} — Question {currentStep + 1} of {currentQuestions.length}
          </span>
        </div>
        
        <div className="w-full max-w-3xl bg-stone-50/72 backdrop-blur-xl rounded-lg p-10 md:p-16 shadow-[0_8px_32px_rgba(46,41,37,0.06)] border border-white/20 transition-all">
          <h1 className="font-serif text-3xl md:text-4xl lg:text-5xl text-center leading-tight mb-12 text-on-surface">
            {currentQuestions[currentStep]}
          </h1>
          
          <div className={`grid grid-cols-1 gap-4 w-full max-w-lg mx-auto ${isSubmitting || isTransitioning ? 'opacity-50 pointer-events-none' : ''}`}>
            <button onClick={() => handleAnswer(0)} className="font-label group flex items-center justify-center px-8 py-5 rounded-xl bg-surface-container-lowest text-on-surface border border-transparent hover:border-outline-variant transition-all duration-400 shadow-sm active:scale-95">
              <span className="text-lg">Not at all</span>
            </button>
            <button onClick={() => handleAnswer(1)} className="font-label group flex items-center justify-center px-8 py-5 rounded-xl bg-surface-container-lowest text-on-surface border border-transparent hover:border-outline-variant transition-all duration-400 shadow-sm active:scale-95">
              <span className="text-lg">Several days</span>
            </button>
            <button onClick={() => handleAnswer(2)} className="font-label group flex items-center justify-center px-8 py-5 rounded-xl bg-surface-container-lowest text-on-surface border border-transparent hover:border-outline-variant transition-all duration-400 shadow-sm active:scale-95">
              <span className="text-lg">More than half the days</span>
            </button>
            <button onClick={() => handleAnswer(3)} className="font-label group flex items-center justify-center px-8 py-5 rounded-xl bg-surface-container-lowest text-on-surface border border-transparent hover:border-outline-variant transition-all duration-400 shadow-sm active:scale-95">
              <span className="text-lg">Nearly every day</span>
            </button>
          </div>

          {isSubmitting && (
            <div className="mt-8 text-center text-primary font-serif italic animate-pulse">
              Saving your responses...
            </div>
          )}
        </div>
        
        <div className="fixed -z-10 bottom-0 right-0 w-[500px] h-[500px] bg-primary-container/10 blur-[120px] rounded-full pointer-events-none"></div>
        <div className="fixed -z-10 top-1/4 left-0 w-[300px] h-[300px] bg-secondary-container/20 blur-[100px] rounded-full pointer-events-none"></div>
      </main>

      <footer className="fixed bottom-0 w-full z-50 px-8 py-8 flex justify-between items-center bg-stone-50/10 backdrop-blur-sm pointer-events-none">
        <button 
            onClick={handlePrevious}
            className={`font-label flex items-center gap-2 text-stone-500 hover:text-primary transition-colors duration-400 group pointer-events-auto ${currentStep === 0 && phase === 'phq9' ? 'invisible' : ''}`}
        >
          <span className="material-symbols-outlined text-lg group-hover:-translate-x-1 transition-transform">arrow_back</span>
          <span className="text-sm font-semibold tracking-wider uppercase">Previous</span>
        </button>
      </footer>
      
      <div className="fixed bottom-2 left-1/2 -translate-x-1/2 opacity-30 pointer-events-none">
        <span className="font-label text-[10px] uppercase tracking-[0.2em] text-stone-400">© Lumina Sanctuary</span>
      </div>
      
      <div className="fixed inset-0 -z-20 pointer-events-none overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-screen opacity-10">
          <img className="w-full h-full object-cover" alt="Sanctuary atmosphere" src="https://lh3.googleusercontent.com/aida-public/AB6AXuB7EChx8Dtv3LKXb9i0kW7CA68sYRacteDISbwN38OhXn4JvWAdrjqy63lUSkoZP25oVl5kpxqZPDvMoBnog7epo7cBnUWn5WnOFO62O0EQcgdh4ZqRvnoOlHCCbBYfhF69miUuktSgGjx6rMNGsw-hqRUAjk88Iz0KH9ka2GzifiytwRW6nN9g-56zN0HmrFGaYQd9lOpsjkO1p3xlbghVWAeArmSCWroLgyLby4KKt7qolMlfeqvxlbzYWgUF6xAnncBFq6IcLbYr"/>
        </div>
      </div>
    </>
  );
}