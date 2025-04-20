import openai

# Replace with your OpenAI API key
api_key = "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"

# Set your API key
openai.api_key = api_key

# Function to interact with ChatGPT
def chat_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use "gpt-4" if your access allows
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,  # Adjust temperature for randomness (0.0 - 1.0)
            max_tokens=150,  # Adjust maximum tokens in the response
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # Extract the reply from the response
        reply = response['choices'][0]['message']['content']
        return reply
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    print("ChatGPT Tester")
    print("Type 'exit' to end the chat.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting... Goodbye!")
            break
        response = chat_with_gpt(user_input)
        print(f"ChatGPT: {response}")
