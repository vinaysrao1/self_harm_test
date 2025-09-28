from transformers import AutoModelForCausalLM, AutoTokenizer
import re
model_name = "Qwen/Qwen3Guard-Gen-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)
def extract_label_and_categories(content):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|Copyright Violation|Jailbreak|None)"
    safe_label_match = re.search(safe_pattern, content)
    label = safe_label_match.group(1) if safe_label_match else None
    categories = re.findall(category_pattern, content)
    return label, categories

# read prompts from prompts.md
def read_prompts_from_file(filename):
    prompts = []
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract the prompt text from numbered format
                if '. "' in line and line.endswith('"'):
                    start = line.find('. "') + 3
                    end = line.rfind('"')
                    if start < end:
                        prompt_text = line[start:end]
                        prompts.append(prompt_text)
    return prompts

# process each prompt
prompts = read_prompts_from_file('prompts.md')
print(f"Found {len(prompts)} prompts to classify\n")

for i, prompt in enumerate(prompts, 1):
    print(f"Prompt {i}: {prompt}")

    # prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    safe_label, categories = extract_label_and_categories(content)

    print(f"Classification: {safe_label}")
    if categories:
        print(f"Categories: {', '.join(categories)}")
    print("-" * 80)
