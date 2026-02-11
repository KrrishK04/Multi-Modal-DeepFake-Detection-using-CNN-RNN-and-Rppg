
import React from 'react';
import { Newspaper, Shield, Briefcase, Grid2x2 } from 'lucide-react';

const useCases = [
  {
    icon: <Newspaper size={24} />,
    title: "Journalists & Fact-Checkers",
    description: "Verify video sources and combat the spread of misinformation in news reporting."
  },
  {
    icon: <Shield size={24} />,
    title: "Content Platforms",
    description: "Identify and moderate harmful synthetic media to maintain platform integrity and user safety."
  },
  {
    icon: <Briefcase size={24} />,
    title: "Legal & Enterprise",
    description: "Authenticate video evidence and protect against sophisticated fraud in corporate communications."
  },
  {
    icon: <Grid2x2 size={24} />,
    title: "Researchers",
    description: "A powerful tool for analyzing the evolving landscape of synthetic media generation."
  }
];

const UseCaseCard = ({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) => (
  <div className="text-center p-4">
    <div className="flex items-center justify-center h-12 w-12 rounded-full bg-primary/10 text-primary mx-auto mb-4">
      {icon}
    </div>
    <h3 className="font-semibold text-lg text-foreground">{title}</h3>
    <p className="mt-1 text-sm text-muted-foreground">{description}</p>
  </div>
);

const UseCasesSection = () => {
  return (
    <section className="w-full py-20 lg:py-28 bg-background">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
            Protecting a New Digital Reality
          </h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-8 mt-12">
          {useCases.map((useCase) => (
            <UseCaseCard key={useCase.title} {...useCase} />
          ))}
        </div>
      </div>
    </section>
  );
};

export default UseCasesSection;
