def generate_hint(error):
    hints = {
        1: "Check arithmetic carefully.",
        2: "Watch signs.",
        3: "Review algebra steps.",
        4: "Complete all steps.",
        5: "Invalid input."
    }
    return hints.get(error, "Good job!")
