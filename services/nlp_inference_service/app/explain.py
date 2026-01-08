import torch
from captum.attr import IntegratedGradients


def compute_token_attributions(model, tokenizer, text, target_label_idx):
    """
    Compute token-level attributions using embeddings (correct for BERT).
    """

    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    # 🔹 Get embeddings instead of token IDs
    embeddings = model.bert.embeddings.word_embeddings(input_ids)

    def forward_func(embeds):
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        )
        return outputs.logits[:, target_label_idx]

    ig = IntegratedGradients(forward_func)

    # Baseline = zero embeddings
    baseline = torch.zeros_like(embeddings)

    attributions, _ = ig.attribute(
        inputs=embeddings,
        baselines=baseline,
        return_convergence_delta=True,
    )

    # Sum attribution across embedding dimension
    token_attributions = attributions.sum(dim=-1).squeeze(0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

    return tokens, token_attributions.detach().cpu().numpy()