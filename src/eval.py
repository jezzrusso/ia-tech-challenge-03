import sys, json, random
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Para baseline, usamos um prompt simples a partir do título.
PROMPT_BASE = (
    "Pergunta: Descreva o produto: {t}\n"
    "Contexto (título do produto): {t}\n"
    "Responda descrevendo o produto de forma fiel ao contexto."
)

def load_test_samples(path, n=30):
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            xs.append(json.loads(line))
    random.Random(42).shuffle(xs)
    return xs[:n] if n else xs

def build_baseline_prompt_from_input(input_text: str) -> str:
    """
    O processed JSONL tem 'input' já completo. Para baseline, simplificamos:
    extraímos o título da linha 'Contexto (título do produto): ...'
    """
    # tenta achar linha com 'Contexto (título do produto): '
    title = None
    for part in input_text.split("\n"):
        key = "Contexto (título do produto): "
        if part.startswith(key):
            title = part[len(key):]
            break
    if not title:
        # fallback: usa tudo mesmo
        title = input_text[:128]
    return PROMPT_BASE.format(t=title)

def predict_many(samples, use_sft=False, max_new_tokens=256):
    base = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(base)
    base_m = AutoModelForSeq2SeqLM.from_pretrained(base)
    model = PeftModel.from_pretrained(base_m, "models/flan-t5-lora") if use_sft else base_m
    outs = []
    for obj in samples:
        gold = obj["output"]
        if use_sft:
            prompt = obj["input"]  # já está no formato da task
        else:
            prompt = build_baseline_prompt_from_input(obj["input"])
        x = tok(prompt, return_tensors="pt")
        y = model.generate(**x, max_new_tokens=max_new_tokens)
        pred = tok.decode(y[0], skip_special_tokens=True)
        outs.append({"prompt": prompt, "gold": gold, "pred": pred})
    return outs

def rougeL(items):
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    vals = [sc.score(it["gold"], it["pred"])["rougeL"].fmeasure for it in items]
    return sum(vals)/len(vals)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"baseline", "sft"}:
        print("Uso: python -m src.eval baseline|sft")
        sys.exit(1)

    mode = sys.argv[1]
    test_path = "data/processed/test.jsonl"
    S = load_test_samples(test_path, n=30)

    preds = predict_many(S, use_sft=(mode == "sft"))
    out_path = f"data/interim/{mode}_preds.jsonl"
    with open(out_path, "w", encoding="utf-8") as w:
        for r in preds:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    score = rougeL(preds)
    print({"mode": mode, "rougeL_f1_mean": score, "n": len(preds), "out": out_path})

    # Amostra qualitativa (primeiros 20) — útil para o vídeo/README
    if mode == "sft":
        with open("data/interim/qual_samples.jsonl", "w", encoding="utf-8") as w:
            for r in preds[:20]:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
