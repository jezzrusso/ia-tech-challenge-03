import sys, json, random, argparse, time
import torch
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Prompt do baseline (extrai apenas o título)
PROMPT_BASE = (
    "Pergunta: Descreva o produto: {t}\n"
    "Contexto (título do produto): {t}\n"
    "Responda descrevendo o produto de forma fiel ao contexto."
)

def load_test_samples(path, n=30, seed=42):
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            xs.append(json.loads(line))
    random.Random(seed).shuffle(xs)
    return xs[:n] if n else xs

def build_baseline_prompt_from_input(input_text: str) -> str:
    title = None
    for part in input_text.split("\n"):
        key = "Contexto (título do produto): "
        if part.startswith(key):
            title = part[len(key):]
            break
    if not title:
        title = input_text[:128]
    return PROMPT_BASE.format(t=title)

def pick_device_and_dtype(force_device: str = "auto"):
    if force_device == "cpu":
        return torch.device("cpu"), None, False
    if torch.cuda.is_available() and force_device in ("auto", "cuda"):
        major, minor = torch.cuda.get_device_capability(0)
        bf16_ok = major >= 8  # Ampere/Ada
        dtype = torch.bfloat16 if bf16_ok else torch.float16
        return torch.device("cuda"), dtype, bf16_ok
    return torch.device("cpu"), None, False

def predict_many(samples, use_sft=False, device=None, dtype=None,
                 max_new_tokens=128, num_beams=4, do_sample=False,
                 temperature=1.0, top_p=1.0):
    base = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(base)
    # carrega base já no dtype preferido (se for CUDA)
    base_m = AutoModelForSeq2SeqLM.from_pretrained(base, torch_dtype=dtype if device.type=="cuda" else None).to(device)
    model = PeftModel.from_pretrained(base_m, "models/flan-t5-lora").to(device) if use_sft else base_m

    model.eval()
    outs = []
    torch.set_grad_enabled(False)

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=dtype)
                    if (device.type == "cuda" and dtype in (torch.float16, torch.bfloat16))
                    else torch.no_grad())
    with autocast_ctx:
        for obj in samples:
            gold = obj["output"]
            prompt = obj["input"] if use_sft else build_baseline_prompt_from_input(obj["input"])
            x = tok(prompt, return_tensors="pt").to(device)
            y = model.generate(
                **x,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            pred = tok.decode(y[0], skip_special_tokens=True)
            outs.append({"prompt": prompt, "gold": gold, "pred": pred})
    return outs

def rougeL(items):
    sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    vals = [sc.score(it["gold"], it["pred"])["rougeL"].fmeasure for it in items]
    return sum(vals)/len(vals) if vals else 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["baseline", "sft"])
    ap.add_argument("--test_path", default="data/processed/test.jsonl")
    ap.add_argument("--n", type=int, default=30, help="n de exemplos (0 = todos)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--save_metrics", action="store_true")
    args = ap.parse_args()

    device, dtype, bf16_ok = pick_device_and_dtype(args.device)
    if device.type == "cuda":
        print(f"[Device] cuda={torch.cuda.is_available()} | name={torch.cuda.get_device_name(0)} | dtype={'bf16' if bf16_ok else 'fp16'}")
    else:
        print("[Device] CPU")

    S = load_test_samples(args.test_path, n=args.n)
    t0 = time.time()
    preds = predict_many(
        S, use_sft=(args.mode=="sft"), device=device, dtype=dtype,
        max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
        do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p
    )
    dt = time.time() - t0

    out_path = f"data/interim/{args.mode}_preds.jsonl"
    with open(out_path, "w", encoding="utf-8") as w:
        for r in preds:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    score = rougeL(preds)
    summary = {"mode": args.mode, "rougeL_f1_mean": score, "n": len(preds), "out": out_path, "elapsed_sec": round(dt,2)}
    print(summary)

    if args.save_metrics:
        with open(f"data/interim/{args.mode}_metrics.json","w",encoding="utf-8") as f:
            json.dump({"rougeL_f1_mean": score, "n": len(preds), "elapsed_sec": dt}, f, ensure_ascii=False, indent=2)

    if args.mode == "sft":
        with open("data/interim/qual_samples.jsonl", "w", encoding="utf-8") as w:
            for r in preds[:20]:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
