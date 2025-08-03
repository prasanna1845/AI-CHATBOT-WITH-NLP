# AI-CHATBOT-WITH-NLP
COMPANY NAME :CODETECH IT

NAME:PRASANNAKUMAR.M

INTERN ID:CT04DZ1481

DURATION:25-07-2025 TO 25-08-2025

MENTOR:NEELA SANTHOSH KUMAR

PROGRAM:

import nltk import pandas as pd import matplotlib.pyplot as plt

nltk.download('punkt') nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

Sample dataset
data = { 'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 'Sales': [200, 150, 300, 280, 500, 700, 650] } df = pd.DataFrame(data)

Predefined responses
greetings = ["hello", "hi", "hey"] exit_words = ["bye", "exit", "quit"]

def preprocess(text): tokens = nltk.word_tokenize(text.lower()) return [lemmatizer.lemmatize(word) for word in tokens]

def chatbot(): print("Hi, I'm DataBot! Ask me about sales. Type 'bye' to exit.") while True: user_input = input("You: ") tokens = preprocess(user_input)

    if any(word in tokens for word in exit_words):
        print("DataBot: Goodbye!")
        break

    elif any(word in tokens for word in greetings):
        print("DataBot: Hello! How can I assist you with the data?")

    elif "chart" in tokens or "graph" in tokens:
        df.plot(x='Day', y='Sales', kind='bar', legend=False)
        plt.title("Sales Chart")
        plt.xlabel("Day")
        plt.ylabel("Sales")
        plt.tight_layout()
        plt.show()
        print("DataBot: Here's your sales chart.")

    elif "average" in tokens:
        avg = df['Sales'].mean()
        print(f"DataBot: The average sale is {avg:.2f}.")

    elif "highest" in tokens or "max" in tokens:
        max_val = df['Sales'].max()
        day = df[df['Sales'] == max_val]['Day'].values[0]
        print(f"DataBot: The highest sale is {max_val} on {day}.")

    elif "summary" in tokens or "summarize" in tokens:
        print("DataBot: Here's the data summary:")
        print(df.describe())

    else:
        print("DataBot: Sorry, I didn't understand that. Try asking about 'chart', 'average', or 'summary'.")
Run chatbot
chatbot()

Description of the code C: reating a simple AI chatbot using NLP, Pandas, and Matplotlib in Python:

🔧 1. Importing Required Libraries

import nltk import pandas as pd import matplotlib.pyplot as plt

nltk: Used for natural language processing (tokenizing and lemmatizing user input).

pandas: For handling and analyzing tabular data.

matplotlib.pyplot: For plotting visual charts (bar chart in this case).

📦 2. Downloading NLTK Resources

nltk.download('punkt') nltk.download('wordnet')

Downloads punkt tokenizer and wordnet lemmatizer dictionary from NLTK.

This is required to break sentences into words and get their base form.

🧠 3. Initializing the Lemmatizer

from nltk.stem import WordNetLemmatizer lemmatizer = WordNetLemmatizer()

lemmatizer reduces words to their root form (e.g., "running" → "run").

📊 4. Sample Sales Dataset

data = { 'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 'Sales': [200, 150, 300, 280, 500, 700, 650] } df = pd.DataFrame(data)

A sample dictionary representing sales for each day of the week.

Converted to a DataFrame for easy analysis and plotting.

💬 5. Predefined User Intent Categories

greetings = ["hello", "hi", "hey"] exit_words = ["bye", "exit", "quit"]

Simple keyword-based intent recognition for greetings and exit.

🧹 6. Preprocessing Function

def preprocess(text): tokens = nltk.word_tokenize(text.lower()) return [lemmatizer.lemmatize(word) for word in tokens]

Converts user input to lowercase.

Tokenizes the text into words.

Lemmatizes each word to its base form.

🤖 7. Chatbot Function

def chatbot(): print("Hi, I'm DataBot! Ask me about sales. Type 'bye' to exit.")

Starts the chatbot interaction.

🚪 Exit Condition

if any(word in tokens for word in exit_words): print("DataBot: Goodbye!") break

Ends chat if user types "bye", "exit", or "quit".

👋 Greetings Handling

elif any(word in tokens for word in greetings): print("DataBot: Hello! How can I assist you with the data?")

Responds with a friendly greeting.

📊 Displaying Sales Chart

elif "chart" in tokens or "graph" in tokens: df.plot(x='Day', y='Sales', kind='bar', legend=False) ...

Shows a bar chart of daily sales using Matplotlib.

➗ Calculating Average Sale

elif "average" in tokens: avg = df['Sales'].mean() print(f"DataBot: The average sale is {avg:.2f}.")

Calculates and displays the mean of sales.

📈 Displaying Highest Sale

elif "highest" in tokens or "max" in tokens: max_val = df['Sales'].max() ...

Finds the maximum sales value and the corresponding day.

🧾 Summary Statistics

elif "summary" in tokens or "summarize" in tokens: print(df.describe())

Prints summary statistics using pandas.DataFrame.describe() (mean, std, min, max, etc.)

🤷 Default Response

else: print("DataBot: Sorry, I didn't understand that...")

Responds when the input doesn't match any known keyword.

▶ 8. Start the Chatbot

chatbot()

Runs the chatbot function in an infinite loop until the user exits.

✅ Overall Functionality

This chatbot:

Understands basic user input through keyword matching.

Uses NLP (tokenizing + lemmatizing) to process natural language.

Analyzes and visualizes sales data using Pandas and Matplotlib.

Does not use machine learning or external models — it's rule-based and ideal for beginners.

OUTPUT:

<img width="1920" height="1080" alt="Image" src="https://github.com/user-attachments/assets/f7f850d0-231f-4fd3-a92a-468703b87c0c" />
