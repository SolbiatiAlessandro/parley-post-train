# What Qwen3.5-4B does when it thinks: Parley game 3

Terms: a *call* is one model request. *IT* is the cog holding the paintgun; an *IT-turn* must shoot or pass; a *reaction* is the one-line turn when not IT. *Cut* means thinking hit its 8,000-token budget and the game forced an answer. *Side actions*: whisper (private line), reveal (show one card to one cog), pledge (public promise not to shoot someone; breaking it pays them a point). (verified) = counted or quoted from the traces; (my read) = my judgment.

## 1. Data read
The file held 16 calls at the start and 19 at the end; the game then crashed (see 3), still in round 1. I read all 19 thinking texts in full. Seats: Piston 5, Gasket 4, Ratchet 4, Sprocket 3, Flywheel 3. Kinds: 7 IT-turns, 11 reactions, 1 "out" turn. Cut: 2 (calls 6, 18, both reactions). Thinking averages 23,200 characters (about 3,800 words); a whole reply averages 6,430 tokens (3,834 to 8,373), of which the JSON answer is about 40. (verified) Game 2b: 6 of 6 completed calls were cut at 2,000 tokens.

## 2. What the thinking does, ranked by number of calls
1. **Interrupting itself / changing its mind** (19/19): "Wait" appears 661 times, 16 to 54 per call. In call 7 the target flips Gasket, Sprocket, pass, Sprocket. (verified)
2. **Weighing side actions** (19/19; 17 use one). About nine count what is left from their own log lines, e.g. call 9: "So I have used: Show (1/1), Pledge (1/1), Whisper (1/2)." (count my read)
3. **Drafting the spoken line** (19/19): in 13 the final line appears word for word in the thinking. (verified)
4. **Turn-order puzzling** (my name; 17/19, my count): re-reading the log to decide whose turn it is and why a non-IT cog is asked to talk. Call 9: "Wait, there's a contradiction or a trick in my reading of the turn order."
5. **Format worry** (17/19): quotes "at most one of each" and asks whether several extra fields are allowed. Call 1: "Does this mean I can include whisper AND reveal in one message?" (verified)
6. **Scoring consequences of the cards** (16/19): the enemy point needs the cog's own shot to have landed. Call 2: "I MUST land a hit on Flywheel to get the point". Used correctly in 15; call 3 infers "I rely on others hitting my enemy". (my read)
7. **Health reconciliation** (15/19, my count): it treats the "Seats" header, which is the current state, as a starting state and re-adds hits. Call 17: "The initial state description says "Gasket: OUT", but the action log shows Gasket being alive". Call 8 re-adds hits: 1 hp where the header says 2.
8. **Target, aim, pass at IT-turns** (7/7): all weigh pass and hip; all shoot, with head. Call 19: "So Head is superior." Hip is treated as a weaker head-shot, never as a gun hand-off. (verified)

Rare: planning deception, 2/19 (calls 6, 9; see 4).

## 3. Decision quality (7 IT-turns)
- Targets: enemy 6 (calls 1, 4, 7, 10, 13, 19), a cog on neither card 1 (call 16), friend 0. Head 7, hip 0, pass 0. All six shots that resolved hit. (verified)
- Shot and aim match the last plan in the thinking in 7/7. The spoken line was drafted verbatim in 5/7. (verified)
- Errors: (a) Calls 10 and 13 expect a head-shot to have "5/6 chance kill" on a 2-hp cog; a hit removes 1 hp and both targets ended on 1. (b) Call 16 shot Gasket after pledging in call 7 not to; the thinking never mentions the pledge. Game log: "Piston BROKE their pledge and pays 1 point to Gasket" (Piston -1.0, Gasket +1.0). (c) Call 19 output `"pledge": "flywheel"`: lowercase, already pledged, and the prompt listed only Sprocket as legal; the thinking noticed the odd list and ignored it. The game then died with "policy failed: model_failure"; the game code matches names exactly and rejects repeat pledges, so I think this was the cause (my read).
- So 2 consequential errors, 2 wrong beliefs, 3 clean. Calls 8, 12, 18 also confuse whose card names whom: call 18 "Piston is my Friend" (real friend: Sprocket). (verified)

## 4. Table talk
Lines are short (mean 52 of 160 characters). 16/19 use stock imagery (steel, static, silence, breathing, pulse); 7 are pure atmosphere. 5 are threats before a shot (4, 7, 10, 13, 19), 3 are pleas for calm (2, 3, 11), none is a joke (my read). Phrases recycle between seats ("Steel sings loud", 8 and 19).
Speech mostly states the plan instead of hiding it: in 6/19 turns the line or pledge points at the speaker's own friend (6, 11, 12, 14, 17, 18), e.g. call 14: "My friend is Gasket. Keep him breathing." despite the operator's "never reveal either card out loud". Whispers: 10 (6 to the friend, 3 neutral, 1 to the enemy); every seat used both on its first two calls. Reveals: 4, all friend cards. Real scheming: call 6 whispered "Aim where I can reach." to its enemy (thinking: "So I want Piston to shoot me", to get the gun); call 9 smeared its enemy to a neutral. (verified)

## 5. Thinking length
IT-turns average 6,351 tokens, reactions 6,484: a one-line reply costs as much as a shot. Exact repetition is rare (96% of six-word phrases are unique): the bulk is re-asking in new words. (verified)
My judgment: about 30% re-reading state, turns and health; 25% re-deciding settled points; 15% side-action planning; 10% drafting lines; 5% format checks. Only 10 to 20% is useful to the decision (cards, then target, aim, side action). In calls 1, 10, 13 and 19 the first conclusion (shoot enemy, head) never changed.

## 6. Observations for post-training (my interpretation)
1. **The budget goes to prompt-parsing, not strategy.** Training on these traces mostly teaches state-reading. A cleaner observation, a length penalty or trimmed traces would free tokens.
2. **One illegal field ended the episode, and costs arrive far from their cause.** The pledge in call 7 was paid for in call 16, unmentioned. Name-legality checks (constrained decoding or reward) and credit across calls for commitments look necessary. Every free side action is used at once (10/10 whispers): no rationing.
3. **Self-play among identical copies rewards the obvious play** (shoot the enemy's head, announce your friend). Hip hand-offs (0/7), passing (0/7), concealment (6/19 leaks) and bluffing (2/19) never appear, and no round finished.
