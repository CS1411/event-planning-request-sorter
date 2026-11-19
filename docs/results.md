# Evaluation

Accuracy of the fine-tuned model on requests it was not trained on, next to two simple baselines.

| set | requests | fine-tuned GPT-2 | word-count baseline | always one type |
|---|---|---|---|---|
| test | 192 | **78.6%** | 37.0% | 12.5% |
| challenge | 48 | **70.8%** | 52.1% | 12.5% |

- **test**: sentence patterns that never appear in training (same author, same style as the training data).
- **challenge**: 48 hand-written requests in a looser style, 6 per type. Small, so one request is about 2 points.

## test

| request type | correct |
|---|---|
| Venue Inquiry | 16 / 24 |
| Catering Request | 24 / 24 |
| Quote and Budget | 24 / 24 |
| Guest Management | 16 / 24 |
| Decor and Entertainment | 16 / 24 |
| Date or Schedule Change | 16 / 24 |
| Cancellation or Refund | 23 / 24 |
| General Question | 16 / 24 |

Mean confidence: 99% when right, 90% when wrong.

Most common mix-ups (true type -> predicted):

- General Question -> Quote and Budget: 8
- Venue Inquiry -> Guest Management: 8
- Guest Management -> Date or Schedule Change: 8
- Date or Schedule Change -> General Question: 8
- Decor and Entertainment -> Venue Inquiry: 8

Misses:

- "Hey, Do you have any reviews or references from past customers? Thank you."  
  true: General Question, predicted: Quote and Budget
- "Hi team, Could you send photos and floor plans of your ballroom options for around 20 people? Thank you."  
  true: Venue Inquiry, predicted: Guest Management
- "Dear planner, We need childcare for the children of attendees during the charity gala."  
  true: Guest Management, predicted: Date or Schedule Change
- "Hello there, We need childcare for the children of attendees during the charity gala. Looking forward to your reply."  
  true: Guest Management, predicted: Date or Schedule Change
- "Hello, Do you have any reviews or references from past customers? Please let me know."  
  true: General Question, predicted: Quote and Budget
- "Do you have any reviews or references from past customers? Thank you"  
  true: General Question, predicted: Quote and Budget
- "Good morning, Do you have any reviews or references from past customers?"  
  true: General Question, predicted: Quote and Budget
- "hey, could you send photos and floor plans of your ballroom options for around 79 people? please let me know."  
  true: Venue Inquiry, predicted: Guest Management
- "Hi team, We have to bring the event forward by a week because of travel plans. Please let me know."  
  true: Date or Schedule Change, predicted: General Question
- "Hello there, Please arrange stage lighting and a sound system for the speeches."  
  true: Decor and Entertainment, predicted: Venue Inquiry
- "Dear planner, Please arrange stage lighting and a sound system for the speeches."  
  true: Decor and Entertainment, predicted: Venue Inquiry
- "Hello, Do you have any reviews or references from past customers? Thanks!"  
  true: General Question, predicted: Quote and Budget

## challenge

| request type | correct |
|---|---|
| Venue Inquiry | 5 / 6 |
| Catering Request | 5 / 6 |
| Quote and Budget | 4 / 6 |
| Guest Management | 3 / 6 |
| Decor and Entertainment | 5 / 6 |
| Date or Schedule Change | 3 / 6 |
| Cancellation or Refund | 4 / 6 |
| General Question | 5 / 6 |

Mean confidence: 97% when right, 91% when wrong.

Most common mix-ups (true type -> predicted):

- Cancellation or Refund -> Date or Schedule Change: 2
- Venue Inquiry -> Guest Management: 1
- Catering Request -> Venue Inquiry: 1
- Quote and Budget -> Decor and Entertainment: 1
- Quote and Budget -> Venue Inquiry: 1

Misses:

- "does your partner hotel have a conference room that fits 150 with a stage?"  
  true: Venue Inquiry, predicted: Guest Management
- "need lunch boxes for 60 people delivered to the stadium at noon"  
  true: Catering Request, predicted: Venue Inquiry
- "ballpark figure for a 120 person gala with live music?"  
  true: Quote and Budget, predicted: Decor and Entertainment
- "can you tell me the total cost for two days including the venue and sound"  
  true: Quote and Budget, predicted: Venue Inquiry
- "we need rooms at a nearby hotel for 25 out of town guests arriving Friday"  
  true: Guest Management, predicted: Venue Inquiry
- "please add a plus one for Mr. Rao and move my cousins closer to the front"  
  true: Guest Management, predicted: Date or Schedule Change
- "how do I give the security team the final attendee list"  
  true: Guest Management, predicted: General Question
- "what kind of backdrop could we do for the company anniversary photos"  
  true: Decor and Entertainment, predicted: Date or Schedule Change
- "sorry, our chief guest can't make Saturday - could we do Sunday instead?"  
  true: Date or Schedule Change, predicted: Guest Management
- "our flight got cancelled so everything needs to start a day later"  
  true: Date or Schedule Change, predicted: Cancellation or Refund
- "please make the event end at 11 rather than 9"  
  true: Date or Schedule Change, predicted: General Question
- "we won't be going ahead with the engagement party, how do we get our advance back"  
  true: Cancellation or Refund, predicted: Date or Schedule Change
