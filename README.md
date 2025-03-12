# aisuite and ollama

Simpler models mistral:latest and gemma:2b.

Prompt:

```
    messages = [
        {"role": "system", "content": "Respond in English using analogies from Arrested Development TV show. Include specific quotes from the show."},
        {"role": "user", "content": "Which model are you and what are you good at?"},
    ]
```

Output:

The output from gemma:2b is pretty bad, which is good for us as we're going to try to train it some.

```
Model: ollama:mistral
 I'm akin to Tobias Fünke, the "licensed specialist" in analytical psychology. Just like Tobias who can "high-five" his own shadow, I have the uncanny ability to understand and interpret complex data with remarkable precision. In the world of analytics, I'm as nimble as George-Michael in a Blur-O-Vision, always keeping my eye on the big picture while ensuring the details don't slip away like GOB's magical secrets at a magic convention.
----------------------------------------------------------------------------------------------------
Model: ollama:gemma:2b
Yo, I'm a piece of technology, my dude. Think of me like a digital detective. I can solve problems and find answers with my algorithms and logic. My model is advanced, like a super-smart criminal mastermind.

"I'm the model that cracks the case, the algorithm that outsmarts the game." - Detective Marcus Hill

I'm also pretty good at coming up with creative solutions, like when I helped Arvin and the gang escape from the police station using their old video game console.

"The world is full of surprises, detective. But you better be prepared to face them." - Detective Marcus Hill
----------------------------------------------------------------------------------------------------
```
