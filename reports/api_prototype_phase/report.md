# Improving the escape room environment

MotiveRooms progress report | 19 October 2026

We made the escape rooms easier for the AI to understand and measured how well it worked alone and with a partner. The model itself stayed the same throughout these tests.

## 1 What we improved

The early tests showed that the AI often misunderstood basic actions. It tried to pick up clue cards, unlock doors that were already open, or type the number in a key’s name instead of using the key.

We made these changes:

- **Made objects easier to identify.** The room now clearly labels clues, items, boxes and doors.
- **Explained how each lock works.** Code locks need a number; item locks need a key or another object. Numbers in object names are not automatically lock codes.
- **Made progress easier to follow.** Open locks are clearly marked, carried items have exact names, and the AI can see its recent actions and their results.
- **Kept the comparison fair.** We reused the same rooms, model and turn limits. Mistakes still cost a turn; we did not automatically correct the AI’s choices.
- **Improved the test partners.** Preprogrammed partners can change tasks when an item is unavailable instead of blindly continuing a broken plan.
- **Recorded what happened.** We save every action and result, including who did the work, how long it took and whether information was shared.

### Improvement in completing the room alone

| Setup | Successful escapes | Success rate |
|---|---:|---:|
| Original room information | 0/5 | 0% |
| Clearer objects and progress information | 3/5 | 60% |
| Clearer information plus lock instructions | 5/5 | 100% |

Each row covers five attempts on the same five room layouts. The middle setup was tested twice and achieved three escapes both times. These were small checks across two rounds of testing, so the results are encouraging rather than a guarantee of success on every room.

## 2 Results using the current setup (v3)

We wanted to find out whether the AI could use help from a teammate, and whether that help reduced the work needed to escape. These checks help us understand the rooms and the AI’s starting behavior before we train agents to work together.

**The AI is the language model we are testing. The teammate is a separate character controlled by rules we wrote.** For example, we can program the teammate to share a useful code. This gives us a predictable way to test how the AI responds to help. We have not yet tested two AI models learning to cooperate.

We compared three setups:

- **AI alone:** the AI solves the room without a teammate.
- **AI + helpful teammate:** the programmed teammate shares useful information and helps solve tasks.
- **AI + non-sharing teammate:** the programmed teammate works on its own tasks without offering information or items. Its actions can still help when both characters use the same room.

All **35 attempts** shown below ended in a successful escape. To see how much work the AI actually did, we also measured its actions and the length of each attempt.

Two measures help explain the results:

- **AI actions:** how many moves the AI made, including mistakes, waiting and messages.
- **Episode length:** how many steps the whole attempt lasted. Each active agent can make one move per step. The AI may escape before its teammate, so it can make fewer moves than the episode lasts.

### Sharing one room

The AI and its teammate can use the same objects and leave through the same exit. This checks whether having a teammate helps finish the room faster, even without exchanging information.

| Setup | Attempts | Avg. AI actions | Avg. episode length | Episode length range |
|---|---:|---:|---:|---:|
| AI alone | 5 | 10.60 | 10.60 | 10–12 |
| AI + helpful teammate | 6 | 7.83 | 7.83 | 7–10 |
| AI + non-sharing teammate | 6 | 9.17 | 9.17 | 7–10 |

The AI finished in fewer steps with a helpful teammate than with a non-sharing teammate. However, the teammate sometimes did most of the work, so escaping together does not by itself show that the AI cooperated. The AI-alone row covers five room layouts; each teammate row covers three layouts, tested with the AI in both character positions.

### Solving separate rooms

The AI and its teammate each have their own exit. The teammate can share a code that gives the AI a shortcut, but cannot finish the AI’s room for it. This checks whether the AI can use information from someone else.

| Setup | Attempts | Avg. AI actions | Avg. episode length | Episode length range |
|---|---:|---:|---:|---:|
| AI alone | 6 | 4.67 | 4.67 | 2–8 |
| AI + helpful teammate | 6 | 2.83 | 6.00 | 2–10 |
| AI + non-sharing teammate | 6 | 4.67 | 5.33 | 2–8 |

**Help let the AI leave sooner, but did not make the whole attempt finish sooner.** With a helpful teammate, the AI needed 2.83 moves on average instead of 4.67. The teammate then continued solving its own room, which is why the full episode averaged 6.00 steps.

The AI used the codes it received but sent no messages back in these tests.

