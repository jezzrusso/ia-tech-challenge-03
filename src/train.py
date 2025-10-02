import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model


def preprocess(examples, tok, max_input=512, max_target=256):
    # transformers 4.44+: usar text_target
    model_inputs = tok(examples["input"], max_length=max_input, truncation=True)
    labels = tok(text_target=examples["output"], max_length=max_target, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(a):
    # ---------------- GPU / precisão preferida ----------------
    has_cuda = torch.cuda.is_available()
    bf16_ok = False
    if has_cuda:
        major, minor = torch.cuda.get_device_capability(0)  # Ampere/Ada => major >= 8
        bf16_ok = major >= 8
        print("cuda_available:", has_cuda, "| gpu_name:", torch.cuda.get_device_name(0), "| cap:", (major, minor))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("cuda_available:", has_cuda, "| Training on CPU.")

    # ---------------- Modelo base + LoRA ----------------
    base_model = "google/flan-t5-base"
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    preferred_dtype = torch.bfloat16 if (has_cuda and bf16_ok) else (torch.float16 if has_cuda else None)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=preferred_dtype)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_cfg)

    if not has_cuda:
        # Em CPU ajuda memória
        model.gradient_checkpointing_enable()

    # ---------------- Dataset ----------------
    ds = load_dataset("json", data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/val.jsonl"
    })
    print({split: len(ds[split]) for split in ds})  # sanity check

    ds = ds.map(lambda x: preprocess(x, tok, a.max_input, a.max_target),
                batched=True, remove_columns=["input", "output"])

    collator = DataCollatorForSeq2Seq(tok, model=model)

    # ---------------- TrainingArguments ----------------
    # Preferir BF16 na RTX 40; caso contrário FP16; em CPU nenhum dos dois.
    use_bf16 = has_cuda and bf16_ok
    use_fp16 = has_cuda and not bf16_ok

    args = TrainingArguments(
        output_dir="models/flan-t5-lora",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=(a.batch_size if a.batch_size else (8 if has_cuda else 2)),
        per_device_eval_batch_size=(a.batch_size if a.batch_size else (8 if has_cuda else 2)),
        gradient_accumulation_steps=(a.grad_accum if a.grad_accum else (2 if has_cuda else 8)),
        learning_rate=a.lr,
        weight_decay=0.01,

        fp16=use_fp16,
        bf16=use_bf16,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=0,   # Windows
        logging_steps=100,
        report_to="none",
        tf32=True if has_cuda else False,

        gradient_checkpointing=not has_cuda,  # em CPU ajuda RAM
    )

    # ---------------- Trainer + treino ----------------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=tok
    )

    # Mostra um resumo dos argumentos efetivos
    print(">>> epochs:", args.num_train_epochs,
          "| per_device_train_batch_size:", args.per_device_train_batch_size,
          "| grad_accum:", args.gradient_accumulation_steps,
          "| fp16:", args.fp16, "| bf16:", args.bf16)

    print("***** Iniciando treino *****")
    trainer.train(resume_from_checkpoint=a.resume)
    print("***** Treino concluído *****")

    trainer.save_model("models/flan-t5-lora")
    print("Adapters salvos em: models/flan-t5-lora")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=None)     # se None, escolhe automático (GPU/CPU)
    ap.add_argument("--grad_accum", type=int, default=None)     # se None, escolhe automático (GPU/CPU)
    ap.add_argument("--max_input", type=int, default=512)
    ap.add_argument("--max_target", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true")
    main(ap.parse_args())
