
import React from 'react';
import FileUploader from '@/components/FileUploader';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';
import ResultsSection from './ResultsSection';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';

interface HeroSectionProps {
  onFileSelect: (file: File | null) => void;
}

const HeroSection = ({ onFileSelect }: HeroSectionProps) => {
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null);
  const [isLoading, setIsLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState<string | null>(null);

  const handleFileSelect = (file: File | null) => {
    setSelectedFile(file);
    if (file) {
      onFileSelect(file);
    }
  };

  const handleAnalyze = async () => {
    if (selectedFile) {
      setIsLoading(true);
      setError(null);
      const formData = new FormData();
      formData.append("video", selectedFile);

      try {
        const response = await fetch("http://127.0.0.1:5000/analyze", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          setResult(data);
        } else {
          setError("Prediction failed");
        }
      } catch (error) {
        setError("Error during prediction");
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <section className="relative w-full min-h-screen flex flex-col items-center justify-center text-center px-4 py-20 hero-gradient">
        <div className="absolute inset-0 bg-black/50" />
        <div className="relative z-10 w-full max-w-4xl animate-fade-in-up">
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white">
                Is It Real? Find Out Now.
            </h1>
            <p className="mt-6 max-w-3xl mx-auto text-lg text-gray-300">
                Our advanced AI analyzes videos for tell-tale signs of deepfakery, from visual artifacts to invisible biological signals.
            </p>

            <div className="mt-12 flex flex-col items-center justify-center space-y-4">
                {isLoading || result ? (
                    <ResultsSection isLoading={isLoading} result={result} />
                ) : (
                    <>
                        <FileUploader onFileSelect={handleFileSelect} selectedFile={selectedFile} />
                        <Button onClick={handleAnalyze} disabled={!selectedFile || isLoading} size="lg" className="font-semibold text-base shadow-lg bg-cyan-500 hover:bg-cyan-600 text-black px-10 py-6 transition-all duration-300 transform hover:scale-105">
                            {isLoading ? <Loader2 className="w-5 h-5 mr-2 animate-spin" /> : null}
                            Analyze Video
                        </Button>
                        <p className="text-xs text-gray-400">Supports .mp4, .mov, .avi up to 50MB.</p>
                        {error && (
                            <Alert variant="destructive" className="max-w-lg mt-4 bg-red-900/50 border-red-500 text-white">
                                <AlertCircle className="h-4 w-4" />
                                <AlertTitle>Error</AlertTitle>
                                <AlertDescription>{error}</AlertDescription>
                            </Alert>
                        )}
                    </>
                )}
            </div>
        </div>
    </section>
  );
};

export default HeroSection;
