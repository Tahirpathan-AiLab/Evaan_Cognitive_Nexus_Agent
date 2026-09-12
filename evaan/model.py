import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_ID, MAX_TURNS_IN_CONTEXT
from persona import build_system_prompt


# 6. LOAD LOCAL MODEL

print("\nLoading Evaan's local model...")
print("Model:", MODEL_ID)
print("Device: CPU")
print(
    "(First run downloads the model. "
    "After that it works from local cache.)\n"
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Tokenizer loaded.")
print("Loading model weights...\n")

# RAM FIX: float32 weights for a 0.5B model cost ~2GB by themselves,
# plus PyTorch/transformers runtime overhead on top of that.
# bfloat16 halves the weight memory (~1GB) and is supported for
# CPU inference in recent torch/transformers versions. We stay
# in torch.float32 ONLY for internal numerically-sensitive ops
# if needed — for a 0.5B causal LM, bfloat16 end-to-end is fine.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

model.eval()

# Make sure autograd machinery never allocates gradient buffers —
# this is inference-only, so this avoids extra memory being held
# around for backward passes we never run.
torch.set_grad_enabled(False)

print("Model loaded successfully.")
print("Evaan is running on CPU.")

# 8. GENERATE RESPONSE

def generate_response(history, mood, relevant_memory=""):
    messages = [
        {"role": "system", "content": build_system_prompt(mood, relevant_memory)}
    ] + history[-MAX_TURNS_IN_CONTEXT:]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,

            max_new_tokens=120,

            temperature=0.3,

            top_p=0.9,

            do_sample=True,

            repetition_penalty=1.1,

            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    reply = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return reply
