"""Milestone 1: define the request types and build a small labelled dataset.

Eight kinds of incoming client request, each with the team that should handle it and how urgent it usually is.
The examples are synthetic: every category has 14 hand-written sentence patterns, filled in with random events,
dates, cities and guest counts, and wrapped in different greetings. Three patterns per category are held back
for testing, so the test set contains sentence shapes the model never saw in training. A separate hand-written
"challenge" set in a different style is kept in dataset/challenge.jsonl.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"

# key -> what planners see. Order matters: it fixes the order of the label tokens.
CATEGORIES = {
    "venue": {"name": "Venue Inquiry", "team": "Venue team", "priority": "normal",
              "next_steps": "Check venue availability and capacity, then send options with photos."},
    "catering": {"name": "Catering Request", "team": "Catering team", "priority": "normal",
                 "next_steps": "Confirm headcount and dietary needs, then send menu options."},
    "quote": {"name": "Quote and Budget", "team": "Sales team", "priority": "high",
              "next_steps": "Prepare an itemised quote against the budget and reply within one day."},
    "guests": {"name": "Guest Management", "team": "Guest services", "priority": "normal",
               "next_steps": "Update the guest list, then arrange rooms, transport or seating."},
    "decor": {"name": "Decor and Entertainment", "team": "Creative team", "priority": "normal",
              "next_steps": "Ask for the theme and mood, then suggest vendors and a mood board."},
    "schedule": {"name": "Date or Schedule Change", "team": "Operations", "priority": "high",
                 "next_steps": "Check every vendor's availability for the new time before confirming."},
    "cancel": {"name": "Cancellation or Refund", "team": "Finance", "priority": "urgent",
               "next_steps": "Review the contract's cancellation terms and reply the same day."},
    "general": {"name": "General Question", "team": "Front desk", "priority": "low",
                "next_steps": "Answer from the FAQ or hand over to the right person."},
}
PRIORITY_ORDER = {"urgent": 0, "high": 1, "normal": 2, "low": 3}


def label_token(key):
    "The special token for a category, for example <Venue Inquiry>."
    return f"<{CATEGORIES[key]['name']}>"


# 14 patterns per category. Patterns 3, 7 and 11 are held out for testing.
TEST_TEMPLATE_IDS = (3, 7, 11)
TEMPLATES = {
    "venue": [
        "Do you have a venue available for a {event} on {date} in {city}?",
        "We're looking for a space that can hold {n} guests for our {event}. What venues do you recommend?",
        "Is the garden terrace available on {date}, and what is its capacity?",
        "Could you send photos and floor plans of your ballroom options for around {n} people?",
        "Can we tour the venue before booking? We are planning a {event} in {city}.",
        "Which venues in {city} allow outdoor ceremonies for a {event}?",
        "Does the venue have parking and step-free access for {n} attendees?",
        "What's the maximum capacity of your rooftop space? It's for a {event}.",
        "I need a location near the city centre for a {event} of about {n} people.",
        "Are there any rooms free on {date} for a small {event}?",
        "Does the hall include a stage, a projector and Wi-Fi for our {event}?",
        "Can you suggest a beachfront or lakeside venue for a {event} in {city}?",
        "Is there an indoor backup option if it rains at the venue on {date}?",
        "How early can we get into the venue to set up for the {event}?",
    ],
    "catering": [
        "We would like a buffet menu for {n} guests at our {event}.",
        "Can the caterer provide vegetarian and vegan options for the {event}?",
        "Do you offer a plated three-course dinner for {n} people?",
        "Please add a dessert table and a coffee station for the {event}.",
        "Several guests have {dietary} allergies. What can the kitchen do?",
        "Can we arrange a tasting of the menu before the {event} on {date}?",
        "I'd like to order canapes and drinks for a cocktail hour for {n} guests.",
        "Is a halal or kosher menu possible for our {event}?",
        "What beverage packages do you offer, including a bar for the {event}?",
        "Could you cater a working lunch for {n} people on {date}?",
        "Please change our dinner choice to a family-style menu for the {event}.",
        "Can you recommend a food truck or street-food style catering for {n} guests?",
        "We need breakfast pastries and fruit delivered at 8am on {date}.",
        "Will there be a birthday cake service included with the {event} menu?",
    ],
    "quote": [
        "Could you send a quote for a {event} with {n} guests in {city}?",
        "Our budget is around {amount}. What can we get for that?",
        "How much does a full-service {event} package cost?",
        "Can you break down the price per person for {n} guests?",
        "We'd like a price estimate before we sign anything for the {event}.",
        "Are there any discounts for booking the {event} early or off-season?",
        "What deposit do you require, and when are payments due?",
        "Please send an itemised invoice covering venue, food and staff.",
        "Is the cost of tables and chairs included in your {event} price?",
        "Can you compare the cheapest and premium options for {n} people?",
        "We need to keep the total under {amount}. Can you adjust the proposal?",
        "What would an all-inclusive {event} cost for a weekend in {city}?",
        "Does your quote include service charge and taxes?",
        "Could you resend the proposal with updated numbers for {n} guests?",
    ],
    "guests": [
        "How do I send invitations and collect RSVPs for the {event}?",
        "Three more guests have confirmed, so we now expect {n} people.",
        "Can you help arrange shuttle buses between the hotel and the venue for {n} guests?",
        "We need wheelchair accessible seating for two relatives.",
        "Please prepare name badges and a seating chart for the {event}.",
        "Some guests are flying in on {date}. Can you book hotel rooms for them?",
        "Can we set up a check-in desk and a guest list for arrivals?",
        "A few attendees need interpreters and a quiet room.",
        "How many guests can we add after the RSVP deadline?",
        "Please send a reminder to everyone who hasn't replied yet.",
        "Can you arrange a welcome desk and parking for VIP guests?",
        "We need childcare for the children of attendees during the {event}.",
        "Two guests cannot attend anymore, please update the headcount to {n}.",
        "Can you set up a table plan so the families are seated together?",
    ],
    "decor": [
        "We want a floral theme with white roses and greenery for the {event}.",
        "Can you book a DJ or a live band for the {event} on {date}?",
        "Is it possible to add a photo booth and balloon arch?",
        "Please arrange stage lighting and a sound system for the speeches.",
        "What decoration packages do you have for a {event} with a rustic theme?",
        "We'd like a magician or a children's entertainer for the afternoon.",
        "Could you set up a projector and screen for our slideshow and video?",
        "Can you recommend a photographer and videographer for the {event}?",
        "I want candles, string lights and centrepieces on every table.",
        "Do you have a color scheme suggestion for a navy and gold {event}?",
        "We need a microphone, a podium and background music during the awards.",
        "Can you build a branded backdrop with our company logo for the {event}?",
        "Is a fireworks display or a laser show possible at the end of the night?",
        "Please add a dance floor and mood lighting for the {event}.",
    ],
    "schedule": [
        "We need to move the {event} from {date} to a later date.",
        "Can we push the start time back by two hours on {date}?",
        "The speakers are delayed, so can we shift the agenda by one day?",
        "Is it possible to reschedule our {event} to the following weekend?",
        "Our dates have changed. Are you free on {date} instead?",
        "Please change the setup time to the morning of {date}.",
        "Can we extend the {event} until midnight instead of ending at 10pm?",
        "We have to bring the event forward by a week because of travel plans.",
        "Can we swap the Saturday booking for a Friday evening?",
        "Please update the schedule: the keynote now starts at 3pm.",
        "Because of the weather we'd like to postpone to {date}.",
        "Could we shorten the programme and finish an hour earlier?",
        "The venue needs to be ready a day earlier, can the date be changed?",
        "Is there any flexibility to move our {event} to next month?",
    ],
    "cancel": [
        "I'm sorry, but we need to cancel the {event} on {date}.",
        "Please cancel our booking and tell me how the refund works.",
        "Can we get our deposit back if we call off the {event}?",
        "What is your cancellation policy for a {event} with {n} guests?",
        "We want to withdraw from the contract and receive a refund.",
        "Due to an emergency, the {event} will not go ahead. Please cancel everything.",
        "I'd like to cancel the catering order and get my money back.",
        "How much would we lose if we cancelled one week before the {event}?",
        "Please terminate the reservation for {date} and confirm in writing.",
        "Our company has cancelled the retreat, so I need a refund of the deposit.",
        "Can you cancel the DJ and the photographer and refund the fees?",
        "We've decided not to hold the {event} after all, what happens to our payment?",
        "Cancel my booking, please. I was charged twice and want the extra payment returned.",
        "Is the deposit refundable if the {event} is called off by the council?",
    ],
    "general": [
        "What are your office opening hours?",
        "Hi, who is my event coordinator and how can I reach them?",
        "Do you also organise events outside {city}?",
        "Can you share examples of events you have organised before?",
        "How far in advance should we get in touch to plan a {event}?",
        "I forgot my login for the client portal. Who can help?",
        "Do you have a brochure or website I can look at?",
        "Do you work with international clients and other languages?",
        "Are you hiring event staff this season?",
        "Is there a number I can call on the day of the event?",
        "Could someone call me back to explain how your service works?",
        "Do you have any reviews or references from past customers?",
        "What is the best way to contact the team outside office hours?",
        "Can you tell me more about what an event planner actually does?",
    ],
}

EVENTS = ["wedding", "birthday party", "corporate retreat", "product launch", "charity gala", "baby shower",
          "graduation party", "conference", "anniversary dinner", "team offsite"]
CITIES = ["Austin", "Lisbon", "Bangalore", "Denver", "Toronto", "Cape Town", "Dublin", "Mumbai", "Seattle", "Berlin"]
DATES = ["June 14", "the first weekend of October", "next March", "12 December", "the 3rd of May",
         "Saturday the 21st", "late August", "New Year's Eve", "the last Friday of the month", "9 November"]
DIETARY = ["nut", "gluten", "shellfish", "dairy", "egg"]
AMOUNTS = ["$5,000", "$12,000", "10,000 euros", "$30,000", "50,000 rupees", "$8,500"]
OPENERS = ["", "", "Hi,", "Hello,", "Good morning,", "Hi team,", "Hello there,", "Dear planner,", "Hey,"]
CLOSERS = ["", "", "", "Thanks!", "Thank you.", "Please let me know.", "Looking forward to your reply.", "Cheers,"]


def make_request(rng, key, template_id):
    "One request: a sentence pattern with random details, a greeting and a sign-off, and a little noise."
    text = TEMPLATES[key][template_id].format(
        event=rng.choice(EVENTS), city=rng.choice(CITIES), date=rng.choice(DATES), n=rng.randint(20, 400),
        dietary=rng.choice(DIETARY), amount=rng.choice(AMOUNTS))
    text = " ".join(part for part in (rng.choice(OPENERS), text, rng.choice(CLOSERS)) if part)
    roll = rng.random()
    if roll < 0.12:
        text = text.lower()           # people type quickly
    elif roll < 0.22:
        text = text.rstrip(".?!")     # and forget punctuation
    return text


def generate_requests(per_template, seed=0):
    """(train, test): per_template examples for every pattern. Patterns in TEST_TEMPLATE_IDS only appear in the test set."""
    rng = random.Random(seed)
    train, test, seen = [], [], set()
    for key, templates in TEMPLATES.items():
        for template_id in range(len(templates)):
            bucket = test if template_id in TEST_TEMPLATE_IDS else train
            made = 0
            while made < per_template:
                text = make_request(rng, key, template_id)
                if text in seen:
                    continue
                seen.add(text)
                bucket.append({"text": text, "label": key, "template": template_id})
                made += 1
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_train():
    return load_jsonl(DATASET_DIR / "train.jsonl")


def load_test():
    return load_jsonl(DATASET_DIR / "test.jsonl")


def load_challenge():
    return load_jsonl(DATASET_DIR / "challenge.jsonl")


def main():
    "Rebuild dataset/train.jsonl and dataset/test.jsonl: `python -m app.data`."
    train, test = generate_requests(per_template=8)
    save_jsonl(train, DATASET_DIR / "train.jsonl")
    save_jsonl(test, DATASET_DIR / "test.jsonl")
    print(f"{len(CATEGORIES)} request types; {len(train)} training and {len(test)} test requests "
          f"({len(TEMPLATES['venue']) - len(TEST_TEMPLATE_IDS)} training patterns and {len(TEST_TEMPLATE_IDS)} "
          f"test-only patterns per type)")
    for ex in train[:4]:
        print(f"  [{CATEGORIES[ex['label']]['name']}] {ex['text']}")


if __name__ == "__main__":
    main()
