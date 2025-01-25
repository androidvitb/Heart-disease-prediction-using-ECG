from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
# Define the metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Update the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_dataset["train"],
    eval_dataset=prepared_dataset["validation"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,  # Add this line
)

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")
