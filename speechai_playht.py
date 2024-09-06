import os
import speech_recognition as sr
from dotenv import load_dotenv
from groq import Groq
from pyht import Client, TTSOptions, Format
import msvcrt  # Only works on Windows, for non-blocking keyboard input
import tempfile
from playsound import playsound
import io

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Initialize PlayHT client
playht_client = Client(
    user_id=os.getenv("PLAY_HT_USER_ID"),
    api_key=os.getenv("PLAY_HT_API_KEY"),
)

# Set TTS options (configure the voice you want to use)
tts_options = TTSOptions(
    voice=os.getenv("AI_VOICE"),
    sample_rate=44100,
    format=Format.FORMAT_MP3,
    speed=1,
)

# Initialize speech recognition
recognizer = sr.Recognizer()

def load_prompt(filename):
    """
    Load the character's prompt from a text file.
    """
    try:
        with open(filename, 'r') as file:
            prompt_content = file.read()
            return {"role": "system", "content": prompt_content}
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Please make sure the file exists.")
        return None

def chat_groq(message, history, system_prompt):
    """
    Communicate with the Groq model using the specified system prompt and conversation history.
    """
    # Combine system prompt with conversation history
    messages = [system_prompt] + history
    messages.append({"role": "user", "content": message})

    response_content = ""
    # Stream response from Groq
    stream = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        max_tokens=150,  # Limiting response length
        temperature=1.0,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += content
    return response_content.strip()

def generate_speech(text):
    """
    Convert text to speech using PlayHT and play the audio using playsound.
    """
    try:
        # Use PlayHT to generate audio chunks
        audio_data = b""
        for chunk in playht_client.tts(text=text, voice_engine="PlayHT2.0-turbo", options=tts_options):
            audio_data += chunk

        # Save audio to a temporary file
        temp_audio_path = tempfile.mktemp(suffix=".mp3")
        with open(temp_audio_path, "wb") as audio_file:
            audio_file.write(audio_data)

        # Play the generated audio using playsound
        playsound(temp_audio_path)

    except Exception as e:
        print(f"Error generating speech: {e}")

def get_audio_input():
    """
    Capture audio input from the user using the microphone.
    """
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... (Press Backspace to stop)")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("Processing audio...")
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            print("Listening timed out. Please try again.")
            return ""
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return ""
        except sr.RequestError:
            print("Sorry, there was an issue with the recognition service.")
            return ""

def continuous_conversation(history, system_prompt):
    """
    Continuous conversation mode: keeps the conversation going until Backspace is pressed.
    """
    print("Continuous conversation mode activated. Press Backspace to stop.")
    while True:
        # Check if the user presses Backspace
        if msvcrt.kbhit() and ord(msvcrt.getch()) == 8:  # ASCII code 8 is Backspace
            print("\nExiting continuous conversation mode.")
            break

        user_message = get_audio_input()
        if not user_message:
            continue  # Skip if audio input fails
        print(f"You said: {user_message}")

        # Convert history to conversation format for Groq
        conversation_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": content} 
                                for i, (user_msg, assistant_msg) in enumerate(history) 
                                for content in (user_msg, assistant_msg)]
        current_prompt = chat_groq(user_message, conversation_history, system_prompt)
        history.append((user_message, current_prompt))
        print(f"Character: {current_prompt}")
        generate_speech(current_prompt)

def main():
    # Ask the user to enter the filename of the character's prompt
    filename = input("Enter the filename of the character prompt (e.g., 'arther.txt'): ").strip()
    system_prompt = load_prompt(filename)
    
    if not system_prompt:
        print("Unable to load the prompt. Exiting.")
        return

    history = []

    print(f"Welcome to the ChatBot! You are now talking to the character defined in '{filename}'.")
    while True:
        print("\nOptions:")
        print("1. Speak to the character (Continuous Mode)")
        print("2. Type your message")
        print("3. Exit")
        choice = input("Choose an option (1/2/3): ")

        if choice == '1':
            continuous_conversation(history, system_prompt)
        elif choice == '2':
            user_message = input("Type your message: ")
            if user_message:
                # Convert history to conversation format for Groq
                conversation_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": content} 
                                        for i, (user_msg, assistant_msg) in enumerate(history) 
                                        for content in (user_msg, assistant_msg)]
                current_prompt = chat_groq(user_message, conversation_history, system_prompt)
                history.append((user_message, current_prompt))
                print(f"Character: {current_prompt}")
                generate_speech(current_prompt)
        elif choice == '3':
            print("Exiting the chat. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
            continue

if __name__ == "__main__":
    main()
