
import TechnologyCard from '@/components/TechnologyCard';
import { Search, Activity, Heart } from 'lucide-react';

const TechnologySection = () => {
  return (
    <section className="w-full py-20 lg:py-28 bg-background/80">
      <div className="container mx-auto px-4">
        <div className="text-center max-w-3xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
            A Multi-Layered Defense System
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
            <TechnologyCard
                icon={<Search size={28} className="text-cyan-400" />}
                title="Phase 1: Spatial Anomaly Detection (CNN)"
                description="We use a ResNeXt50 network to scan every frame for pixel-level artifacts, unnatural textures, and inconsistent lighting that are often invisible to the human eye."
            />
            <TechnologyCard
                icon={<Activity size={28} className="text-cyan-400" />}
                title="Phase 2: Temporal Inconsistency Detection (RNN)"
                description="An LSTM network analyzes the video's timeline to detect unnatural motion, flawed facial expressions, and inconsistent blinking patterns that break the laws of natural human behavior."
            />
            <TechnologyCard
                icon={<Heart size={28} className="text-cyan-400" />}
                title="Phase 3: Biological Signal Verification (rPPG)"
                description="Our key differentiator. We analyze for a real human heartbeat by detecting subtle skin color changes from blood flow. Most AI generators cannot fake this, making it a robust sign of authenticity."
            />
        </div>
      </div>
    </section>
  );
};

export default TechnologySection;
