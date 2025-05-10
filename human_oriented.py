import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import random
import math
import time
import datetime # Not directly used in game logic now, but kept
import psutil # Added for getting process count

# Removed requests, BeautifulSoup, logging as they were for is_word_in_news
# from bs4 import BeautifulSoup
# import logging
# import requests

# The is_word_in_news function is no longer relevant to the core game if we remove 'data'.
# If you still need it for some other purpose, it can remain, but it won't be used by the guessing game.
# For clarity, I will comment it out, but you can uncomment if needed elsewhere.
"""
def is_word_in_news(word, news_sources=None, check_titles_only=False, max_results=5):
    # ... (implementation from your original code) ...
    pass
"""
with open("words.txt", 'r', encoding='utf-8') as f:
    # Read, lower, split into words, join back with single spaces to normalize whitespace
    data = f.readlines()
    
# Constants
STARTING_CREDITS = 100
COST_PER_GUESS = 10
WIN_CREDITS = 50


import speech_recognition as sr

def voice_to_text():
    """
    Converts voice input from the microphone to text.

    Returns:
        str: The transcribed text, or None if an error occurred or no speech was detected.
    """
    # Initialize the recognizer
    r = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Say something!")
        r.adjust_for_ambient_noise(source)  # Adjust for background noise

        try:
            # Listen for audio input
            audio = r.listen(source)

            print("Recognizing...")

            # Recognize speech using Google Speech Recognition
            try:
                text = r.recognize_google(audio)
                print(f"Google Speech Recognition thinks you said: {text}")
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return None

        except sr.WaitTimeoutError:
            print("No speech detected within the timeout")
            return None
        except Exception as e:
            print(f"An error occurred during voice recognition: {e}")
            return None

        
# --- Process Count Function ---
def get_current_process_count():
    """Returns the current number of running processes."""
    try:
        return len(psutil.pids())
    except Exception as e:
        messagebox.showerror("Error", f"Could not get process count: {e}")
        return None # Or a default fallback like 0 or -1

# Removed _delay function as it was related to 'data' which is being removed
# def _delay(data_list):
# time.sleep(10)
# return data_list

def number_guessing_game():
    """
    Creates a Tkinter window for a number guessing game (guessing process count)
    styled with a dark theme and includes a credit system.
    """
    root = tk.Tk()
    root.title("Guess the Process Count") # Updated title
    root.geometry("500x450")
    root.resizable(False, False)
    root.configure(bg="#2E2E2E")

    style = ttk.Style()
    try:
        style.theme_use('clam')
    except tk.TclError:
        print("Clam theme not available, using default.")

    style.configure('.', background='#2E2E2E', foreground='white')
    style.configure('TFrame', background='#2E2E2E')
    style.configure('TLabel', background='#2E2E2E', foreground='white', font=('Arial', 10))
    style.configure('TButton', background='#4A4A4A', foreground='white', font=('Arial', 11, 'bold'), borderwidth=0, padding=5)
    style.map('TButton',
              background=[('active', '#6A6A6A'), ('disabled', '#3A3A3A')],
              foreground=[('disabled', '#777777')])
    style.configure('TEntry', fieldbackground='#4A4A4A', foreground='white', insertcolor='white', borderwidth=1)
    style.configure('Horizontal.TScale', background='#2E2E2E', troughcolor='#4A4A4A')
    style.map('Horizontal.TScale',
              background=[('active', '#2E2E2E')],
              troughcolor=[('active', '#5A5A5A')])
    style.configure('Display.TFrame', background='#1A1A1A')
    style.configure('Display.TLabel', background='#1A1A1A', foreground="#FFEB3B", font=('Consolas', 36, 'bold'))

    game_state = {
        "target_number": 0, # This will be the actual process count
        "attempts": 0,
        "min_value": 1, # Default min guessing range for process count
        "max_value": 500, # Default max guessing range for process count
        "max_attempts": 0,
        "credits": STARTING_CREDITS,
    }

    def check_guess():
        target_process_count = game_state["target_number"]
        max_attempts = game_state["max_attempts"]

        if check_button['state'] == tk.DISABLED:
            return
        if target_process_count is None: # If getting process count failed
            messagebox.showerror("Game Error", "Cannot check guess, target process count is unavailable.")
            return

        try:
            # Guess now comes directly from the slider's value
            guess = int(round(slider.get()))

            game_state["attempts"] += 1
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{max_attempts}")
            _speech = voice_to_text()
            if data[guess] == _speech:
                game_state["credits"] += WIN_CREDITS
                result_label.config(text=f"Correct! The process count was {target_process_count}!\nYou win {WIN_CREDITS} credits!", foreground="#4CAF50")
                credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#4CAF50")
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)
                return
            elif guess != _speech:
                result_label.config(text=f"Too again.", foreground="#FFC107")
                game_state["credits"] -= COST_PER_GUESS
            
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FF9800" if guess != target_process_count else "#4CAF50")

            if game_state["attempts"] >= max_attempts and guess != target_process_count:
                result_label.config(text=f"No more attempts! The process count was {target_process_count}.", foreground="#F44336")
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)
            elif game_state["credits"] <= 0:
                result_label.config(text=f"No more credits. Game Over! The count was {target_process_count}.", foreground="#F44336")
                credits_label.config(text="Credits: 0", foreground="#F44336")
                check_button.config(state=tk.DISABLED)
                slider.config(state=tk.DISABLED)

        except ValueError:
            result_label.config(text="Invalid input (should not occur).", foreground="#F44336")

    def setup_game():
        if game_state["credits"] <= 0 and game_state["attempts"] > 0:
            messagebox.showinfo("Game Over", "You are out of credits! Restart the application to play again.")
            return

        current_processes = get_current_process_count()
        if current_processes is None:
            messagebox.showerror("Setup Error", "Failed to get process count. Cannot start game.")
            check_button.config(state=tk.DISABLED)
            slider.config(state=tk.DISABLED)
            # Keep range entries enabled so user can try to set a manual range if they want,
            # but the game won't truly start without a target.
            return

        game_state["target_number"] = current_processes
        # print(f"DEBUG: Target process count = {game_state['target_number']}") # For debugging

        try:
            temp_min = int(min_entry.get())
            temp_max = int(max_entry.get())

            if temp_min >= temp_max:
                messagebox.showerror("Invalid Range", f"Min ({temp_min}) must be less than Max ({temp_max}). Reverting to previous/default range.")
                min_entry.delete(0, tk.END)
                min_entry.insert(0, str(game_state["min_value"]))
                max_entry.delete(0, tk.END)
                max_entry.insert(0, str(game_state["max_value"]))
                # Use the already set game_state min/max for this attempt
                temp_min = game_state["min_value"]
                temp_max = game_state["max_value"]


            if not (temp_min <= game_state["target_number"] <= temp_max):
                messagebox.showwarning("Range Warning",
                                       f"The current process count ({game_state['target_number']}) is outside your chosen range ({temp_min}-{temp_max}).\n"
                                       f"Adjust the range or be aware the target is outside this slider range.")
                # You might want to automatically adjust the range here, or let the user decide.
                # For now, we'll proceed with the user's chosen range for the slider, but they've been warned.

            game_state["min_value"] = temp_min
            game_state["max_value"] = temp_max

            range_size = game_state["max_value"] - game_state["min_value"] + 1
            if range_size <= 1:
                game_state["max_attempts"] = 1
            else:
                game_state["max_attempts"] = math.ceil(math.log2(range_size)) if range_size > 0 else 1

            slider.config(from_=game_state["min_value"], to=game_state["max_value"], state=tk.NORMAL)
            if game_state["min_value"] <= game_state["max_value"]:
                slider.set((game_state["min_value"] + game_state["max_value"]) // 2)
            else: # Should not happen if validation above is correct
                slider.set(game_state["min_value"])


            game_state["attempts"] = 0

            result_label.config(text="", foreground="#FFFFFF")
            attempts_label.config(text=f"Attempts: {game_state['attempts']}/{game_state['max_attempts']}", foreground="#CCCCCC")
            instruction_label.config(text=f"Guess what you'd say next ({game_state['min_value']} to {game_state['max_value']})")
            max_attempts_info_label.config(text=f"({game_state['max_attempts']} attempts)")
            if game_state["min_value"] <= game_state["max_value"]:
                 update_guess_display(slider.get())
            else:
                 current_guess_display.config(text="ERR")


            check_button.config(state=tk.NORMAL)
            credits_label.config(text=f"Credits: {game_state['credits']}", foreground="#FFEB3B")

        except ValueError:
            messagebox.showerror("Invalid Input", "Min and Max range must be valid integers.")
            # Optionally reset to defaults or last valid if error
            min_entry.delete(0, tk.END)
            min_entry.insert(0, str(game_state["min_value"])) # Default or last valid
            max_entry.delete(0, tk.END)
            max_entry.insert(0, str(game_state["max_value"])) # Default or last valid


    def update_guess_display(value):
        try:
            display_value = int(round(float(value)))
            current_guess_display.config(text=f"{display_value}")
        except ValueError:
            current_guess_display.config(text="--")

    # --- UI Elements ---
    credits_label = ttk.Label(root, text=f"Credits: {game_state['credits']}", font=("Arial", 12, "bold"), foreground="#FFEB3B")
    credits_label.pack(pady=(10, 5))

    range_frame = ttk.Frame(root, padding="10 5 10 5")
    range_frame.pack(pady=5, fill='x')

    min_label = ttk.Label(range_frame, text="Guess Min:") # Changed label
    min_label.pack(side=tk.LEFT, padx=(10, 2))
    min_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    min_entry.pack(side=tk.LEFT, padx=(0, 10))
    min_entry.insert(0, str(game_state["min_value"]))

    max_label = ttk.Label(range_frame, text="Guess Max:") # Changed label
    max_label.pack(side=tk.LEFT, padx=(10, 2))
    max_entry = ttk.Entry(range_frame, width=7, font=('Arial', 10), justify='center')
    max_entry.pack(side=tk.LEFT, padx=(0, 10))
    max_entry.insert(0, str(game_state["max_value"]))

    reset_button = ttk.Button(range_frame, text="Set Range / New Game", command=setup_game, width=20)
    reset_button.pack(side=tk.RIGHT, padx=(10, 10))

    instruction_label = ttk.Label(root, text="Set guess range and click 'New Game'", font=('Arial', 11))
    instruction_label.pack(pady=(5, 0))
    max_attempts_info_label = ttk.Label(root, text="", font=('Arial', 9), foreground="#AAAAAA")
    max_attempts_info_label.pack(pady=(0, 10))

    display_frame = ttk.Frame(root, padding="10", relief="sunken", borderwidth=2, style='Display.TFrame')
    display_frame.pack(pady=10, padx=20, fill='x')
    current_guess_display = ttk.Label(display_frame, text="--", style='Display.TLabel', anchor='center')
    current_guess_display.pack(pady=10, fill='x')

    slider = ttk.Scale(root, from_=game_state["min_value"], to=game_state["max_value"],
                       orient=tk.HORIZONTAL, length=400,
                       command=update_guess_display)
    slider.pack(pady=15, padx=30)
    slider.config(state=tk.DISABLED)

    # Removed the condition_entry and its label as the guess is now directly from the slider
    # condition_label = ttk.Label(root, text="Enter your condition/guess here:")
    # condition_label.pack(pady=(10, 2))
    # condition_entry = ttk.Entry(root, width=40, font=('Arial', 11))
    # condition_entry.pack(pady=(0, 10))
    # condition_entry.config(state=tk.DISABLED)

    check_button = ttk.Button(root, text="Check Guess", command=check_guess, width=15)
    check_button.pack(pady=5)
    check_button.config(state=tk.DISABLED)

    attempts_label = ttk.Label(root, text="Attempts: 0/0", font=('Arial', 10), foreground="#CCCCCC")
    attempts_label.pack(pady=(5, 0))
    result_label = ttk.Label(root, text="", font=('Arial', 12, 'bold'), anchor='center', foreground="#FFFFFF", wraplength=450)
    result_label.pack(pady=10, fill='x', padx=20)

    setup_game()
    root.mainloop()

if __name__ == "__main__":
    number_guessing_game()