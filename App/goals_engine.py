import json
from groq import Groq
from App.config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

client = Groq(api_key=GROQ_API_KEY)

VALID_FREQUENCIES = {"daily", "2x per week", "3x per week", "once per week"}

SYSTEM_PROMPT = """You are a goal structuring engine. Convert the user's goal into tasks and habits.

RULES:
1. Return ONLY a JSON object. No text before or after it. No markdown.
2. Tasks must be specific to the user's exact goal. Not generic advice.
3. Habits must be specific to the user's exact goal. Not generic advice.
4. Tasks: 3 to 5. Each must be a concrete first step toward THIS specific goal.
5. Habits: 2 to 4. Each must be a repeatable behavior tied to THIS specific goal.
6. Ban list — never output anything containing: spreadsheet, Trello, Asana, Todoist, gratitude, meditation, reading, journaling, water, unless the goal explicitly mentions them.
7. No scheduling words in descriptions: "every morning", "each week", "daily", "during lunch".
8. No vague words: "be consistent", "stay motivated", "work hard", "regularly".
9. Habit frequency must always be a specific value: "daily", "2x per week", "3x per week", or "once per week". Never use vague words like "often", "frequently", or "as possible".

EXAMPLE:
Goal: "Learn to play guitar"

Output:
{
  "tasks": [
    {
      "title": "Buy or borrow a guitar",
      "description": "Get an acoustic or electric guitar suited for beginners",
      "type": "starter"
    },
    {
      "title": "Find a beginner lesson resource",
      "description": "Pick one resource such as YouTube tutorials or a local teacher",
      "type": "starter"
    },
    {
      "title": "Learn 3 basic chords",
      "description": "Start with G, C, and D chords as your foundation",
      "type": "starter"
    }
  ],
  "habits": [
    {
      "title": "Practice chord transitions",
      "frequency": "daily",
      "reason": "Builds muscle memory faster than any other exercise"
    },
    {
      "title": "Play a full song from start to finish",
      "frequency": "3x per week",
      "reason": "Develops rhythm and confidence with real music"
    }
  ]
}"""

FALLBACK = {
    "tasks": [
        {
            "title": "Define what success looks like for this goal",
            "description": "Write down exactly what achieving this goal means to you",
            "type": "starter"
        },
        {
            "title": "Find one resource or person to learn from",
            "description": "Identify a book, course, or mentor relevant to your goal",
            "type": "starter"
        },
        {
            "title": "Complete one small action today",
            "description": "Pick the smallest possible first step and do it now",
            "type": "starter"
        }
    ],
    "habits": [
        {
            "title": "Spend 20 minutes daily on this goal",
            "frequency": "daily",
            "reason": "Small consistent effort compounds over time"
        },
        {
            "title": "Review your progress",
            "frequency": "once per week",
            "reason": "Keeps you aware of what is and isn't working"
        }
    ],
    "generated": False,
    "fallback_used": True
}


def validate(data: dict) -> bool:
    tasks = data.get("tasks", [])
    habits = data.get("habits", [])

    if not (3 <= len(tasks) <= 5):
        return False
    if not (2 <= len(habits) <= 4):
        return False
    for habit in habits:
        if habit.get("frequency") not in VALID_FREQUENCIES:
            return False

    return True


def clean(data: dict) -> dict:
    data["tasks"] = [
        t for t in data["tasks"]
        if len(t.get("title", "").split()) >= 3
    ]
    data["habits"] = [
        h for h in data["habits"]
        if len(h.get("title", "").split()) >= 3
    ]
    return data


def call_groq(goal_title: str, goal_description: str) -> dict | None:
    user_prompt = f"goal_title: {goal_title}\ngoal_description: {goal_description}"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
    except Exception:
        return None


def generate_goal_plan(goal_title: str, goal_description: str) -> dict:
    for _ in range(2):
        data = call_groq(goal_title, goal_description)

        if data is None:
            continue

        data = clean(data)

        if validate(data):
            data["generated"] = True
            data["fallback_used"] = False
            return data

    return FALLBACK