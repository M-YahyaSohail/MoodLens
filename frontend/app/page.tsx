"use client"

import { useCallback, useMemo, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Spinner } from "@/components/ui/spinner"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AspectRatio } from "@/components/ui/aspect-ratio"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"

const EMOTION_COLORS: Record<string, string> = {
  anger: "#d9534f",
  disgust: "#9467bd",
  fear: "#6a51a3",
  joy: "#e8a020",
  neutral: "#7f8fa6",
  sadness: "#4a90d9",
  surprise: "#e84393",
}

const SENTIMENT_COLORS: Record<string, string> = {
  negative: "#d9534f",
  neutral: "#7f8fa6",
  positive: "#2ecc71",
}

const FRIENDLY_ERRORS: Record<number, string> = {
  400: "This file doesn't look like a supported video format (MP4, MOV, AVI, MKV, WebM…).",
  413: "The video is too large for the server's upload limit (200 MB). Try a shorter or more compressed clip.",
  422: "The video is longer than the server allows. Try a shorter clip.",
  429: "Too many requests — the server's rate limit was hit. Wait a minute and try again.",
  503: "The server is busy analyzing another video right now. Try again in a moment.",
  504: "Analysis timed out. Try a shorter video.",
}

const EMOTION_EMOJIS: Record<string, string> = {
  anger: "😠",
  disgust: "🤢",
  fear: "😨",
  joy: "😊",
  neutral: "😐",
  sadness: "😢",
  surprise: "😲",
}

const SENTIMENT_EMOJIS: Record<string, string> = {
  negative: "😞",
  neutral: "😐",
  positive: "😄",
}

interface EmotionPrediction {
  label: string
  confidence: number
}

interface SentimentPrediction {
  label: string
  confidence: number
}

interface Utterance {
  start_time: number
  end_time: number
  text: string
  emotions: EmotionPrediction[]
  sentiments: SentimentPrediction[]
}

interface AnalyzeResponse {
  utterances: Utterance[]
}

type View = "idle" | "analyzing" | "results" | "error"

function percent(value: number) {
  return Math.round(value * 100)
}

function formatTime(seconds: number) {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, "0")}`
}

export default function Page() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [view, setView] = useState<View>("idle")
  const [file, setFile] = useState<File | null>(null)
  const [utterances, setUtterances] = useState<Utterance[]>([])
  const [error, setError] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0] ?? null
      setFile(f)
      setError(null)
      setView("idle")
      setUtterances([])
    },
    [],
  )

  const buttonLabel = file ? `Change file (${file.name})` : "Choose video file"

  const handleChooseFile = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!file) return
    setIsAnalyzing(true)
    setError(null)
    setView("analyzing")
    setUtterances([])
    try {
      const formData = new FormData()
      formData.append("video", file)

      const res = await fetch(`${API_URL}/analyze/video`, {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        let detail = FRIENDLY_ERRORS[res.status] ?? `Request failed (${res.status})`
        try {
          const json = await res.json()
          detail = (json.detail as string) ?? detail
        } catch {
          // ignore parse errors, use fallback detail
        }
        throw new Error(detail)
      }

      const data: AnalyzeResponse = await res.json()
      const segments = data.utterances ?? []
      setUtterances(segments)
      if (segments.length === 0) {
        setView("results")
      } else {
        setView("results")
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error")
      setView("error")
    } finally {
      setIsAnalyzing(false)
    }
  }, [file])

  const handleRetry = useCallback(() => {
    setError(null)
    setView("idle")
  }, [])

  const handleAnalyzeAnother = useCallback(() => {
    setFile(null)
    setUtterances([])
    setError(null)
    setView("idle")
  }, [])

  const summary = useMemo(() => {
    const emotionCounts: Record<string, number> = {}
    const sentimentCounts: Record<string, number> = {}
    for (const u of utterances) {
      u.emotions.forEach((e) => {
        emotionCounts[e.label] = (emotionCounts[e.label] ?? 0) + 1
      })
      u.sentiments.forEach((s) => {
        sentimentCounts[s.label] = (sentimentCounts[s.label] ?? 0) + 1
      })
    }
    const topEmotion =
      Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "neutral"
    const topSentiment =
      Object.entries(sentimentCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ??
      "neutral"
    return { topEmotion, topSentiment }
  }, [utterances])

  const isLoading = isAnalyzing && view === "analyzing"

  return (
    <>
      <div className="flex max-w-5xl flex-col gap-6 px-6 py-8 mx-auto">
      {/* Header */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <span className="text-2xl font-heading font-semibold tracking-tighter uppercase">
            MoodLens
          </span>
          <span className="text-xl">🎬</span>
        </div>
        <p className="text-sm text-muted-foreground">
          Multimodal emotion &amp; sentiment analysis — upload a video clip and every
          spoken utterance gets classified by a local deep model.
        </p>
      </div>
      <Separator />

      {/* Settings / health card (Streamlit sidebar equivalent) */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Settings</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <div className="flex flex-col gap-1.5">
            <Label className="text-xs">API address</Label>
            <div className="flex items-center gap-2 text-sm">
              <span className="font-mono text-muted-foreground">{API_URL}</span>
              <span className="ml-auto text-xs font-medium">
                {isLoading ? (
                  <span className="flex items-center gap-1">
                    <Spinner className="size-3.5" />
                    analyzing…
                  </span>
                ) : (
                  <span className="flex items-center gap-1">
                    <span
                      className="h-1.5 w-1.5 rounded-full bg-emerald-500"
                      aria-label="API online"
                    />
                    online
                  </span>
                )}
              </span>
            </div>
          </div>
          <Separator />
          <div className="text-xs text-muted-foreground leading-relaxed">
            <p>
              Limits: uploads ≤ <b>200 MB</b>, max video length <b>10 min</b>, one
              analysis at a time.
            </p>
            <p>
              Start the API with: <span className="font-mono">uv run python main.py</span>
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Upload section */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Upload video</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-3">
          <div className="flex items-center gap-3">
            <Label htmlFor="video-input">Video file</Label>
            <input
              ref={fileInputRef}
              id="video-input"
              type="file"
              accept="video/*"
              disabled={isLoading}
              onChange={handleFileChange}
              className="sr-only"
            />
            <Button
              onClick={handleChooseFile}
              disabled={isLoading}
              className="w-full md:w-auto"
            >
              {buttonLabel}
            </Button>
          </div>

          {file && (
            <>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="text-xs font-medium">{file.name}</span>
              </div>

              <AspectRatio ratio={16 / 9} className="border border-border rounded-none bg-muted overflow-hidden">
                <video
                  src={URL.createObjectURL(file)}
                  controls
                  className="size-full object-cover"
                />
              </AspectRatio>

              <Button
                onClick={handleAnalyze}
                disabled={isLoading || !file}
                className="w-full md:w-auto"
              >
                {isLoading ? "Analyzing…" : "Analyze video"}
              </Button>
            </>
          )}
        </CardContent>
      </Card>

      {/* Loading */}
      {isLoading && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Analyzing video</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center justify-center gap-3 py-10">
            <Spinner className="size-8" />
            <p className="text-sm text-muted-foreground">
              Transcribing speech and running multimodal inference… this can take a
              while for longer videos.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {view === "results" && utterances.length > 0 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle className="text-sm">Results</CardTitle>
            <CardDescription className="text-xs">
              {utterances.length} segment{utterances.length !== 1 ? "s" : ""} detected
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
            <div className="grid grid-cols-2 gap-3">
              <Card className="p-3 border">
                <CardTitle className="text-xs">Dominant emotion</CardTitle>
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{EMOTION_EMOJIS[summary.topEmotion]}</span>
                  <Badge variant="outline" style={{ color: EMOTION_COLORS[summary.topEmotion] }}>
                    {summary.topEmotion}
                  </Badge>
                </div>
              </Card>
              <Card className="p-3 border">
                <CardTitle className="text-xs">Dominant sentiment</CardTitle>
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{SENTIMENT_EMOJIS[summary.topSentiment]}</span>
                  <Badge
                    variant="outline"
                    style={{ color: SENTIMENT_COLORS[summary.topSentiment] }}
                  >
                    {summary.topSentiment}
                  </Badge>
                </div>
              </Card>
            </div>
            <Separator />
            {utterances.map((u, i) => (
              <Card key={i} className="overflow-hidden border">
                <CardHeader className="flex flex-row items-center justify-between gap-2 pb-1">
                  <div className="flex items-center gap-2">
                    <Badge
                      variant="outline"
                      style={{
                        color: EMOTION_COLORS[u.emotions[0]?.label],
                        backgroundColor: "rgba(0,0,0,0.05)",
                      }}
                    >
                      {u.emotions[0]?.label}
                    </Badge>
                    <Badge
                      variant="outline"
                      style={{
                        color: SENTIMENT_COLORS[u.sentiments[0]?.label],
                        backgroundColor: "rgba(0,0,0,0.05)",
                      }}
                    >
                      {u.sentiments[0]?.label}
                    </Badge>
                  </div>
                  <CardDescription className="text-xs">
                    {formatTime(u.start_time)} → {formatTime(u.end_time)}
                  </CardDescription>
                  <span className="text-xs text-muted-foreground font-mono">
                    {percent(u.emotions[0]?.confidence ?? 0)}% top
                  </span>
                </CardHeader>
                {u.text && (
                  <CardContent className="pt-1 pb-1">
                    <p className="text-sm italic text-muted-foreground">
                      “{u.text}”
                    </p>
                  </CardContent>
                )}
                <CardContent className="pt-1 pb-1">
                  <div className="grid gap-2">
                    <div>
                      <Label className="text-xs">Emotions</Label>
                      <div className="grid gap-1.5">
                        {u.emotions.map((e, j) => (
                          <div key={j} className="flex items-center gap-2">
                            <span className="text-lg">{EMOTION_EMOJIS[e.label]}</span>
                            <span className="text-xs font-medium w-16 text-left">
                              {e.label}
                            </span>
                            <Progress
                              value={percent(e.confidence)}
                              className="h-2 flex-1"
                            />
                            <span className="text-[10px] text-muted-foreground w-10 text-right">
                              {percent(e.confidence)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Label className="text-xs">Sentiments</Label>
                      <div className="grid gap-1.5">
                        {u.sentiments.map((s, j) => (
                          <div key={j} className="flex items-center gap-2">
                            <span className="text-lg">{SENTIMENT_EMOJIS[s.label]}</span>
                            <span className="text-xs font-medium w-16 text-left">
                              {s.label}
                            </span>
                            <Progress
                              value={percent(s.confidence)}
                              className="h-2 flex-1"
                            />
                            <span className="text-[10px] text-muted-foreground w-10 text-right">
                              {percent(s.confidence)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </CardContent>
          <CardFooter>
            <Button variant="outline" onClick={handleAnalyzeAnother} className="w-full md:w-auto">
              Analyze another video
            </Button>
          </CardFooter>
        </Card>
      )}

      {/* Empty results notice */}
      {view === "results" && utterances.length === 0 && (
        <Alert>
          <AlertTitle>No speech segments detected</AlertTitle>
          <AlertDescription>
            No speech segments were detected in this clip, so there is nothing to
            analyze. Try a video with audible dialogue.
          </AlertDescription>
        </Alert>
      )}

      {/* Error */}
      {view === "error" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Analysis failed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">{error}</p>
            <p className="text-sm text-muted-foreground mt-2">
              Make sure the FastAPI backend is running on
              <span className="font-mono">{API_URL}</span> and that
              <span className="font-mono">MOODLENS_ALLOWED_ORIGINS</span> includes this
              page&apos;s origin (e.g. <span className="font-mono">http://localhost:3000</span>).
            </p>
          </CardContent>
          <CardFooter>
            <Button onClick={handleRetry}>Try again</Button>
          </CardFooter>
        </Card>
      )}
    </div>

    {/* Full-width footer */}
    <footer className="w-full border-t border-background/50 bg-background/80 backdrop-blur-sm px-6 py-6">
      <div className="max-w-5xl mx-auto">
        <Separator />
        <Card>
          <CardContent className="py-4 text-center">
            <p className="text-xs text-muted-foreground mb-1">
              🏆MoodLens - {" "}
              <span className="font-medium">PAI (Programming for AI)</span>
            </p>
            <p className="text-xs text-muted-foreground mb-1">
              <span className="font-medium">🏦FAST NUCES Karachi</span>
            </p>
            <p className="text-xs text-muted-foreground">
              Team members:
              <br />
              <span className="font-medium">😎Taher Mustansir</span> — 25K-0119
              <br />
              <span className="font-medium">🥸Abdullah Ahmad</span> — 25K-0084
              <br />
              <span className="font-medium">🥴M. Yahya Sohail</span> — 25K-0078
            </p>
          </CardContent>
        </Card>
      </div>
    </footer>
    </>
  )
}
