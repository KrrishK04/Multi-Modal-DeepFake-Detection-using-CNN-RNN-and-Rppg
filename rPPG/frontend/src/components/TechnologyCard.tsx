
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface TechnologyCardProps {
  icon: React.ReactNode;
  title: string;
  description: string;
}

const TechnologyCard = ({ icon, title, description }: TechnologyCardProps) => {
  return (
    <Card className="glassmorphism text-center hover:-translate-y-2 transition-transform duration-300 h-full flex flex-col p-4">
      <CardHeader className="items-center">
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-cyan-900/50">
          {icon}
        </div>
        <CardTitle className="mt-4 text-lg font-semibold text-white">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex-grow">
        <p className="text-sm text-gray-400">{description}</p>
      </CardContent>
    </Card>
  );
};

export default TechnologyCard;
