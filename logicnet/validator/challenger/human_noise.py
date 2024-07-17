import random


def get_condition():
    profiles = [
        "math enthusiast",
        "math student",
        "research mathematician",
        "math teacher",
        "theoretical physicist",
        "engineer",
        "student",
        "teacher",
        "researcher",
        "physicist",
        "scientist",
        "mathematician",
        "data scientist",
        "math tutor",
        "math hobbyist",
        "data analyst",
        "data engineer",
        "data enthusiast",
        "data student",
        "data teacher",
        "data researcher",
    ]

    mood = [
        "curious",
        "puzzled",
        "eager",
        "analytical",
        "determined",
        "excited",
    ]

    tone = [
        "inquisitive",
        "thoughtful",
        "meticulous",
        "enthusiastic",
        "serious",
        "playful",
    ]

    return {
        "profile": random.choice(profiles),
        "mood": random.choice(mood),
        "tone": random.choice(tone),
    }
