
import { AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import CircularProgress from '@/components/CircularProgress';
import { cn } from '@/lib/utils';

interface ResultsSectionProps {
  isLoading: boolean;
  result: any;
}

const ResultsSection = ({ isLoading, result }: ResultsSectionProps) => {
  if (isLoading) {
    return (
      <div className="w-full max-w-lg h-[288px] flex flex-col items-center justify-center glassmorphism p-8 space-y-4">
        <Loader2 className="w-16 h-16 text-cyan-400 animate-pulse-strong" />
        <h3 className="text-xl font-semibold text-center text-white">Scanning...</h3>
        <p className="text-sm text-center text-muted-foreground">Our model is performing a multi-layered analysis.</p>
      </div>
    );
  }

  if (!result) return null;

  const isDeepfake = result.prediction === 'FAKE';
  const color = isDeepfake ? 'hsl(var(--destructive))' : 'hsl(var(--primary))';
  const title = isDeepfake ? "Deepfake Detected" : "Likely Authentic";
  const Icon = isDeepfake ? AlertCircle : CheckCircle;
  const explanation = isDeepfake
    ? "Analysis revealed temporal inconsistencies and a lack of authentic rPPG biological signals, strongly indicating AI manipulation."
    : "The video's spatial and temporal data appear consistent with real-world recordings. No significant manipulation artifacts were found.";

  return (
    <div className={cn("w-full max-w-lg h-[288px] flex flex-col items-center justify-center glassmorphism p-8 space-y-3 animate-fade-in-up", isDeepfake ? "border-destructive" : "border-primary")}>
        <CircularProgress progress={result.confidence} color={color} />
        <div className="text-center">
            <h3 className="text-2xl font-bold flex items-center justify-center gap-2" style={{ color }}>
                <Icon className="w-7 h-7" />
                {title}
            </h3>
            <p className="mt-2 text-sm text-muted-foreground">{explanation}</p>
        </div>
    </div>
  );
};

export default ResultsSection;
