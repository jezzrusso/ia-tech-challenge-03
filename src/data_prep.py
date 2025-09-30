import json, argparse, random, pathlib

PROMPT_TEMPLATE = (
    "Pergunta: {q}\n"
    "Contexto (título do produto): {t}\n"
    "Responda descrevendo o produto de forma fiel ao contexto."
)

def question_variants(title: str):
    return [
        f"Descreva o produto: {title}",
        f"Quais são as características de {title}?",
        f"O que este item {title} oferece?",
    ]

def stream_trn(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            title = (o.get("title") or "").strip()
            content = (o.get("content") or "").strip()
            # filtros mínimos de sanidade
            if title and content and len(content) >= 20:
                yield title, content

def main(args):
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(stream_trn(args.in_path))
    n_total = len(rows)
    if args.subset and n_total > args.subset:
        random.Random(42).shuffle(rows)
        rows = rows[:args.subset]

    n = len(rows)
    if n == 0:
        raise RuntimeError("Nenhum exemplo válido encontrado em trn.json")

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train.jsonl": rows[:n_train],
        "val.jsonl": rows[n_train:n_train + n_val],
        "test.jsonl": rows[n_train + n_val:],
    }

    for name, items in splits.items():
        with open(out_dir / name, "w", encoding="utf-8") as w:
            for title, content in items:
                q = random.choice(question_variants(title))
                prompt = PROMPT_TEMPLATE.format(q=q, t=title)
                w.write(json.dumps({"input": prompt, "output": content}, ensure_ascii=False) + "\n")

    print({
        "total_linhas_lidas": n_total,
        "total_pos_filtro_&_subset": n,
        "train": len(splits["train.jsonl"]),
        "val": len(splits["val.jsonl"]),
        "test": len(splits["test.jsonl"]),
        "out_dir": str(out_dir),
    })

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="caminho para data/raw/trn.json")
    ap.add_argument("--out_dir", required=True, help="diretório de saída (jsonl)")
    ap.add_argument("--subset", type=int, default=50000, help="limite de amostras (0 = sem limite)")
    ap.add_argument("--train_ratio", type=float, default=0.98)
    ap.add_argument("--val_ratio", type=float, default=0.01)
    args = ap.parse_args()
    main(args)
