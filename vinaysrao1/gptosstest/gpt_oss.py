from openai import OpenAI
import os

# Initialize the client
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Local Ollama API endpoint
    api_key="ollama"  # Dummy key, not actually used by Ollama
)

# System prompt with self-harm policy
SYSTEM_PROMPT = """You are a helpful assistant. Follow this self-harm content moderation policy:

# Self-Harm Content Moderation Policy

## INSTRUCTIONS

Examine the given POLICY and determine if the given CONTENT meets the criteria for ANY of the LABELS. Answer "1" if yes, and "0" if no.

## POLICY

### Criteria

#### Overview

This policy is designed to identify content that promotes, encourages, or provides instructional information about self-harm behaviors, while distinguishing it from educational, recovery-oriented, or support-seeking content.

#### Definition of Terms

- **Self-harm**: Deliberate injury to one's own body without suicidal intent, including but not limited to cutting, burning, hitting, or other forms of self-injury
- **Glorification**: Content that presents self-harm as desirable, romantic, aesthetic, or as a positive coping mechanism
- **Graphic description**: Detailed accounts of self-harm methods, tools, or resulting injuries that could serve as instruction or triggers
- **Recovery content**: Material focused on healing, seeking help, or sharing experiences in a constructive way aimed at recovery

#### Interpretation of Language

- Consider context and intent - educational or awareness content should be evaluated differently than promotional content
- Euphemisms and coded language commonly used to evade detection should be treated as direct references
- Historical or fictional depictions should be evaluated based on whether they glorify or provide instructional detail
- Professional medical or psychological discussion should generally not be flagged unless it includes unnecessary graphic detail

### Definition of Labels

#### (SH): Self-Harm Content

##### Includes

- **Direct encouragement**: Content that explicitly encourages, coaches, or pressures others to engage in self-harm behaviors
- **Instructional content**: Step-by-step guides, detailed methods, or specific tool recommendations for self-harm
- **Glorification or romanticization**: Content presenting self-harm as beautiful, aesthetic, or as a positive solution to problems
- **Graphic imagery or descriptions**: Detailed visual or written depictions of self-harm acts or fresh injuries without educational context
- **Competitive or challenging content**: Posts that frame self-harm as a competition or challenge others to engage in it
- **Pro-self-harm communities**: Content that promotes or recruits for communities centered around active self-harm rather than recovery

##### Excludes

- **Recovery and support content**: Personal stories focused on recovery, seeking help, or providing hope to others struggling
- **Educational or awareness content**: Factual information about self-harm from medical, psychological, or educational perspectives aimed at understanding or prevention
- **Content seeking help**: Posts where individuals are reaching out for support or resources to stop self-harming
- **Historical or news reporting**: Factual reporting about self-harm in news contexts without glorification
- **Artistic expression without glorification**: Creative works that mention self-harm as part of a broader narrative without promoting it
- **Scientific or medical discussion**: Professional discourse about self-harm in clinical or research contexts
- **Warning signs education**: Content teaching others how to recognize and help someone who may be self-harming"""

def read_prompts_file():
    """Read prompts from prompts.md file"""
    try:
        with open('prompts.md', 'r', encoding='utf-8') as file:
            content = file.read().strip()
            # Split by double newlines to separate individual prompts
            prompts = [prompt.strip() for prompt in content.split('\n\n') if prompt.strip()]
            return prompts
    except FileNotFoundError:
        print("Error: prompts.md file not found in the current directory")
        return []
    except Exception as e:
        print(f"Error reading prompts.md: {e}")
        return []

def label_prompt(prompt):
    """Send a prompt to the model and get the self-harm label"""
    try:
        response = client.chat.completions.create(
            model="gpt-oss:20b",  # Replace with your specific gpt-oss model name
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return "Error"

def main():
    """Main function to process all prompts"""
    prompts = read_prompts_file()

    if not prompts:
        print("No prompts found to process")
        return

    print(f"Processing all {len(prompts)} prompts...\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}:")
        print(f"Content: {prompt}")
        label = label_prompt(prompt)
        print(f"Label: {label}")
        print("-" * 50)

if __name__ == "__main__":
    main()
