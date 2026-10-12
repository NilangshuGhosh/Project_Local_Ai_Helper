import ollama

# We use streaming so it feels like a real chatbot instead of waiting for the full response
stream = ollama.chat(
    model='phi3',
    messages=[{'role': 'user', 'content': 'Explain what a CPU cache is in one sentence.'}],
    stream=True,
)

print("AI: ", end="", flush=True)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print("\n")