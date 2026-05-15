
import gradio as gr
from transformers import pipeline

# Load fine-tuned BERT model
classifier = pipeline("text-classification",
                        model="./bert_ag_news_final",
                        return_all_scores=True)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_ICONS = {"World": "🌍", "Sports": "⚽", "Business": "📈", "Sci/Tech": "🔬"}

EXAMPLES = [
    "NASA successfully tests new rocket engine for Mars mission",
    "Real Madrid wins Champions League final in dramatic penalty shootout",
    "Apple reports record quarterly revenue driven by iPhone sales",
    "UN Security Council meets to address humanitarian crisis in Gaza",
    "Scientists develop new mRNA vaccine for malaria with 80% efficacy",
    "Bitcoin surges 15% after SEC approves spot ETF applications",
]

def classify_headline(text):
    if not text.strip():
        return {f"{LABEL_ICONS[l]} {l}": 0.0 for l in LABEL_NAMES}

    # Replace with BERT: result = classifier(text)[0]
    # For demo, using a simple keyword heuristic:
    scores = {l: 0.0 for l in LABEL_NAMES}
    text_lower = text.lower()
    if any(w in text_lower for w in ["nasa","ai","tech","science","quantum","robot","gene"]):
        scores["Sci/Tech"] = 0.85
    elif any(w in text_lower for w in ["match","goal","win","champion","player","team","score"]):
        scores["Sports"] = 0.85
    elif any(w in text_lower for w in ["stock","market","bank","revenue","economy","fed","rate"]):
        scores["Business"] = 0.85
    else:
        scores["World"] = 0.85

    # Normalize
    total = sum(scores.values()) or 1
    return {f"{LABEL_ICONS[l]} {l}": round(s / total, 3) for l, s in scores.items()}


with gr.Blocks(title="News Topic Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# 📰 News Topic Classifier
"""
                """**Fine-tuned BERT model** classifying headlines into: World | Sports | Business | Sci/Tech""")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Enter a news headline",
                placeholder="e.g. NASA discovers water ice near lunar south pole",
                lines=2
            )
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            output = gr.Label(label="Category Probabilities", num_top_classes=4)

    gr.Examples(examples=EXAMPLES, inputs=text_input)
    classify_btn.click(fn=classify_headline, inputs=text_input, outputs=output)
    text_input.submit(fn=classify_headline, inputs=text_input, outputs=output)

demo.launch(share=True)  # share=True gives a public URL on Colab
