import FaqSection from "./sections/FaqSection";
import HeroSection from "./sections/HeroSection";
import ResultsSection from "./sections/ResultsSection";
import TechnologySection from "./sections/TechnologySection";
import UseCasesSection from "./sections/UseCasesSection";
import { useState } from "react";

const Index = () => {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileSelect = async (file: File | null) => {
    if (file) {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("video", file);

      try {
        const response = await fetch("http://127.0.0.1:5000/analyze", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          setResult(data);
        } else {
          console.error("Prediction failed");
        }
      } catch (error) {
        console.error("Error during prediction:", error);
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-col gap-16 md:gap-32">
      <HeroSection onFileSelect={handleFileSelect} />
      <ResultsSection isLoading={isLoading} result={result} />
      <TechnologySection />
      <UseCasesSection />
      <FaqSection />
    </div>
  );
};

export default Index;