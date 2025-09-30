import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

PROMPT = (
    "Pergunta: {q}\n"
    "Contexto (título do produto): {t}\n"
    "Responda descrevendo o produto de forma fiel ao contexto."
)

def run_inference(title: str, question: str, max_new_tokens=256):
    base = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(base)
    base_m = AutoModelForSeq2SeqLM.from_pretrained(base)
    model = PeftModel.from_pretrained(base_m, "models/flan-t5-lora")

    prompt = PROMPT.format(q=question, t=title)
    x = tok(prompt, return_tensors="pt")
    y = model.generate(**x, max_new_tokens=max_new_tokens)
    return tok.decode(y[0], skip_special_tokens=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--question", default="Descreva o produto")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    print(run_inference(args.title, args.question, args.max_new_tokens))
