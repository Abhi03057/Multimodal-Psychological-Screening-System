"use client";

import React, { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

export default function LiveSession() {
  const router = useRouter();
  
  // APIs and State
  const [questions, setQuestions] = useState<any[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [sessionHistory, setSessionHistory] = useState<any[]>([]);
  const [lastAnalysis, setLastAnalysis] = useState<any>(null);
  
  // Media State
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  
  const [isRecording, setIsRecording] = useState(false);
  const [isFinishing, setIsFinishing] = useState(false);
  const [timer, setTimer] = useState(0);

  // Fetch Questions
  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const res = await fetch("http://localhost:8000/live/questions");
        const data = await res.json();
        setQuestions(data.questions);
      } catch (err) {
        console.error("Failed to fetch questions:", err);
      }
    };
    fetchQuestions();
  }, []);

  // Timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording && !isFinishing) {
      interval = setInterval(() => setTimer(t => t + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording, isFinishing]);

  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  // Start Camera & MediaRecorder
  useEffect(() => {
    let stream: MediaStream | null = null;
    let timerID: NodeJS.Timeout;

    const startRecording = async () => {
      try {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        } catch (mediaErr: any) {
          console.warn("Could not start video source. Falling back to audio only...", mediaErr);
          stream = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
        }
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        let options: any = { mimeType: 'audio/webm' };
        if (!MediaRecorder.isTypeSupported('audio/webm')) {
          options = MediaRecorder.isTypeSupported('audio/mp4') ? { mimeType: 'audio/mp4' } : undefined;
        }

        // Extract only the audio tracks to prevent NotSupportedError when using an audio-only MIME type
        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length === 0) {
          throw new Error("No audio tracks found in the media stream.");
        }
        const audioStream = new MediaStream(audioTracks);

        const mediaRecorder = new MediaRecorder(audioStream, options);
        mediaRecorderRef.current = mediaRecorder;
        
        mediaRecorder.ondataavailable = async (e) => {
          if (e.data.size > 0 && !isFinishing) {
            await analyzeChunk(e.data);
          }
        };

        try {
          // Chunk every 5 seconds
          mediaRecorder.start(5000);
        } catch (err) {
          console.warn("Timeslicing start failed, falling back to manual requestData", err);
          mediaRecorder.start();
          timerID = setInterval(() => {
            if (mediaRecorder.state === "recording") {
              try { mediaRecorder.requestData(); } catch (e) {}
            }
          }, 5000);
        }
        setIsRecording(true);

      } catch (err) {
        console.error("Camera/Mic access denied:", err);
      }
    };

    if (questions.length > 0 && !isFinishing) {
      startRecording();
    }

    return () => {
      if (timerID) clearInterval(timerID);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (stream) {
        stream.getTracks().forEach(t => t.stop());
      }
    };
  }, [questions, currentQuestionIndex]);

  // Analyze Chunk
  const analyzeChunk = async (audioBlob: Blob) => {
    let frameBlob: Blob | null = null;
    
    // Capture Frame
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        const canvas = canvasRef.current;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          frameBlob = await new Promise<Blob>((resolve) => canvas.toBlob(b => resolve(b as Blob), 'image/jpeg'));
        }
      }
    }

    const formData = new FormData();
    const phq9 = sessionStorage.getItem("phq9_score") || "0";
    const gad7 = sessionStorage.getItem("gad7_score") || "0";
    const qt = questions[currentQuestionIndex]?.category || "unknown";

    formData.append("audio_file", audioBlob, "chunk.webm");
    if (frameBlob) {
      formData.append("frame_file", frameBlob, "frame.jpg");
    }
    formData.append("phq9_score", phq9);
    formData.append("gad7_score", gad7);
    formData.append("question_type", qt);

    try {
      const res = await fetch("http://localhost:8000/live/analyze", {
        method: "POST",
        body: formData
      });
      const result = await res.json();
      setLastAnalysis(result);
      setSessionHistory(prev => [...prev, result]);
    } catch(err) {
      console.error("Analyze chunk error", err);
    }
  };

  const handleNext = async () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
    } else {
      await endSession();
    }
  };

  const endSession = async () => {
    setIsFinishing(true);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    const payload = {
      phq9_score: parseInt(sessionStorage.getItem("phq9_score") || "0"),
      gad7_score: parseInt(sessionStorage.getItem("gad7_score") || "0"),
      history: sessionHistory
    };

    try {
      const res = await fetch("http://localhost:8000/live/end-session", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      
      // Save report to session storage for the report page
      sessionStorage.setItem("lumina_report", JSON.stringify(data));
      router.push("/report");
    } catch(err) {
      console.error("Failed to end session:", err);
      setIsFinishing(false);
    }
  };

  const currentQ = questions[currentQuestionIndex] || { prompt: "Loading question..." };
  const progressPercent = questions.length > 0 ? ((currentQuestionIndex + 1) / questions.length) * 100 : 0;

  return (
    <>
      <nav className="fixed top-0 w-full z-50 bg-[#fcf9f6]/72 dark:bg-[#2E2925]/72 backdrop-blur-xl shadow-[0_8px_32px_rgba(46,41,37,0.06)]">
        <div className="flex justify-between items-center px-12 py-6 w-full max-w-screen-2xl mx-auto">
          <div className="text-2xl font-serif italic text-[#715b3a] dark:text-[#c4a882]">Lumina</div>
          <div className="hidden md:flex gap-8 items-center">
            <Link className="text-[#436275] hover:text-[#715b3a] font-sans text-sm tracking-wide transition-opacity duration-300" href="/">Sanctuary</Link>
            <Link className="text-[#436275] hover:text-[#715b3a] font-sans text-sm tracking-wide transition-opacity duration-300" href="/report">Progress</Link>
            <Link className="text-[#715b3a] dark:text-[#c4a882] font-semibold border-b-2 border-[#c4a882] pb-1 font-serif italic tracking-tight" href="#">Breathing</Link>
          </div>
          <div className="flex items-center gap-4">
            <button className="material-symbols-outlined text-primary">account_circle</button>
          </div>
        </div>
      </nav>

      <main className="flex-grow pt-28 pb-32 px-12 max-w-screen-2xl mx-auto w-full flex flex-col gap-8">
        <header className="w-full space-y-4">
          <div className="flex justify-between items-end">
            <div className="space-y-1">
              <p className="text-secondary font-label text-sm uppercase tracking-[0.2em]">Live Session</p>
              <h1 className="font-serif italic text-3xl text-primary">Inner Discovery</h1>
            </div>
            <div className="text-right">
              <span className="text-primary-container font-serif italic text-lg">
                Question {currentQuestionIndex + 1} of {questions.length || '?'}
              </span>
              <span className="text-on-surface-variant font-body text-sm ml-2 opacity-60">
                ({Math.round(progressPercent)}% complete)
              </span>
            </div>
          </div>
          <div className="w-full h-1 bg-surface-container rounded-full overflow-hidden">
            <div className="h-full bg-primary-container transition-all duration-700 ease-in-out" style={{ width: `${progressPercent}%`}}></div>
          </div>
        </header>

        <div className="flex-grow grid grid-cols-1 md:grid-cols-12 gap-12">
          {/* Left Column: Vision & Emotion */}
          <section className="md:col-span-5 flex flex-col gap-6">
            <div className="relative group rounded-lg overflow-hidden shadow-[0_8px_32px_rgba(46,41,37,0.06)] aspect-video md:aspect-auto md:flex-grow bg-surface-container-highest">
              
              <video 
                ref={videoRef} 
                className={`w-full h-full object-cover ${!isRecording ? 'hidden' : ''} ${isFinishing ? 'opacity-50' : ''}`}
                autoPlay 
                muted 
                playsInline
                style={{ transform: 'scaleX(-1)' }} // mirror
              ></video>
              <canvas ref={canvasRef} className="hidden" />

              {!isRecording && !isFinishing && (
                 <div className="w-full h-full flex items-center justify-center text-sm font-label text-secondary lowercase">
                   Loading camera...
                 </div>
              )}
              
              <div className="absolute top-6 left-6 flex items-center gap-2 px-3 py-1.5 glass-panel rounded-full border border-outline-variant/15">
                <span className={`w-2 h-2 rounded-full ${isRecording && !isFinishing ? 'bg-error animate-pulse' : 'bg-secondary'}`}></span>
                <span className="text-xs font-label text-on-surface tracking-wider">
                  {isFinishing ? "Processing..." : "Camera Active"}
                </span>
              </div>
            </div>
            
            {/* Emotion Chips */}
            <div className="flex flex-wrap gap-3 items-center">
              <span className="text-on-surface-variant text-sm font-label mr-2">Observed State:</span>
              <div className="px-4 py-1.5 bg-tertiary-fixed rounded-full flex items-center gap-2">
                <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>spa</span>
                <span className="text-on-tertiary-fixed text-sm font-body capitalize">
                  {lastAnalysis?.facial_emotion || "Analyzing..."}
                </span>
              </div>
              <div className="px-4 py-1.5 bg-secondary-fixed rounded-full flex items-center gap-2">
                <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>mic</span>
                <span className="text-on-secondary-fixed text-sm font-body capitalize">
                  {lastAnalysis?.audio_emotion || "Listening..."}
                </span>
              </div>
            </div>
          </section>

          {/* Right Column: Inquiry & Metrics */}
          <section className="md:col-span-7 flex flex-col gap-8">
            <div className="bg-surface-container-lowest p-10 rounded-lg shadow-[0_8px_32px_rgba(46,41,37,0.06)] flex-grow flex flex-col justify-center border border-outline-variant/10">
              <span className="material-symbols-outlined text-primary-container text-4xl mb-6">format_quote</span>
              <h2 className="font-serif text-3xl md:text-3xl leading-snug text-primary max-w-xl transition-opacity">
                {currentQ.prompt}
              </h2>
            </div>
            
            {/* Audio Metrics */}
            <div className="glass-panel p-8 rounded-lg border border-outline-variant/15">
              <div className="grid grid-cols-3 gap-6">
                <div className="space-y-3">
                  <label className="block text-[10px] uppercase tracking-widest text-on-surface-variant/60 font-label">Speech Energy</label>
                  <div className="text-sm text-primary font-body uppercase">
                    {lastAnalysis?.audio_energy || "-"}
                  </div>
                </div>
                <div className="space-y-3">
                  <label className="block text-[10px] uppercase tracking-widest text-on-surface-variant/60 font-label">Vocal Pitch</label>
                  <div className="text-sm text-primary font-body uppercase">
                    {lastAnalysis?.audio_stability || "-"}
                  </div>
                </div>
                <div className="space-y-3">
                  <label className="block text-[10px] uppercase tracking-widest text-on-surface-variant/60 font-label">Risk Profile</label>
                  <div className="flex items-center gap-2 text-primary font-body text-sm uppercase">
                    {lastAnalysis?.risk_level === "high" ? <span className="w-1.5 h-1.5 rounded-full bg-error"></span> : <span className="w-1.5 h-1.5 rounded-full bg-primary"></span>}
                    {lastAnalysis?.risk_level || "-"}
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>

      <footer className="fixed bottom-0 w-full z-50 glass-panel border-t border-outline-variant/10">
        <div className="max-w-screen-2xl mx-auto px-12 py-6 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3 px-5 py-2.5 bg-surface-container-high rounded-full">
              <span className="material-symbols-outlined text-primary text-xl">timer</span>
              <span className="font-body font-medium text-primary tracking-wider">{formatTime(timer)}</span>
            </div>
            <p className="text-on-surface-variant font-serif italic text-lg opacity-70 hidden md:block">
              {isFinishing ? "Generating editorial report..." : "Take your time. No wrong answers."}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <button 
                onClick={handleNext} 
                disabled={isFinishing || questions.length === 0}
                className="px-8 py-4 bg-primary text-on-primary rounded-xl font-label text-sm uppercase tracking-widest hover:bg-primary/90 transition-all active:scale-95 flex items-center gap-3 disabled:opacity-50"
            >
              {currentQuestionIndex === questions.length - 1 ? "Finish Session" : "Next Question"}
              {isFinishing ? <span className="material-symbols-outlined text-lg animate-spin">refresh</span> : <span className="material-symbols-outlined text-lg">arrow_forward</span>}
            </button>
          </div>
        </div>
      </footer>
    </>
  );
}