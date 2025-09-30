import argparse
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
    base_model = "google/flan-t5-base"

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    # LoRA leve
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/val.jsonl"
    })
    ds = ds.map(lambda x: preprocess(x, tok, a.max_input, a.max_target),
                batched=True, remove_columns=["input", "output"])

    collator = DataCollatorForSeq2Seq(tok, model=model)

    # Estratégias explícitas (agora suportadas nessa versão)
    args = TrainingArguments(
        output_dir="models/flan-t5-lora",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        weight_decay=0.01,
        fp16=a.fp16,                 # ative se sua GPU suportar
        bf16=False,
        logging_steps=100,
        report_to="none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=tok
    )

    trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model("models/flan-t5-lora")
    print("Treino concluído. Adapters salvos em: models/flan-t5-lora")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_input", type=int, default=512)
    ap.add_argument("--max_target", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="ative se sua GPU suportar")
    main(ap.parse_args())
