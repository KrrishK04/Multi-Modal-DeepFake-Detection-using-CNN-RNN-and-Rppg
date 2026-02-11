
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"

const faqs = [
    {
        question: "How accurate is this model?",
        answer: "Our model achieved 92.4% accuracy on a comprehensive validation set, including datasets like FaceForensics++ and Celeb-DF. While highly accurate, no model is perfect, and results should be considered a strong indicator rather than absolute proof."
    },
    {
        question: "Is my data safe?",
        answer: "Absolutely. We respect your privacy. Uploaded videos are processed in memory and permanently deleted from our servers immediately after the analysis is complete. We do not store your data."
    },
    {
        question: "What can't it detect?",
        answer: "This model is specialized for video deepfakes (face swaps, puppet-master). It may not detect audio-only fakes, simple video edits (like speeding up/slowing down), or out-of-frame manipulation."
    },
    {
        question: "Do you offer an API for developers?",
        answer: "We are currently developing a robust API for integration into other platforms. If you are interested in early access, please contact us."
    }
]

const FaqSection = () => {
    return (
        <section className="w-full py-20 lg:py-28 bg-background">
            <div className="container mx-auto px-4 max-w-3xl">
                <div className="text-center mb-12">
                    <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-foreground">
                        Have Questions? We Have Answers.
                    </h2>
                </div>
                <Accordion type="single" collapsible className="w-full">
                    {faqs.map((faq, index) => (
                        <AccordionItem value={`item-${index+1}`} key={index} className="border-white/10">
                            <AccordionTrigger className="text-left hover:no-underline">{faq.question}</AccordionTrigger>
                            <AccordionContent className="text-muted-foreground">
                                {faq.answer}
                            </AccordionContent>
                        </AccordionItem>
                    ))}
                </Accordion>
            </div>
        </section>
    )
}

export default FaqSection;
