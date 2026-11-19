# Model Card: PawPal+ AI Coach

## 1. Model overview

| | |
|---|---|
| **System name** | PawPal+ AI Coach |
| **Base project** | PawPal+ (Module 2): a Streamlit pet-care scheduler with a rule-based `Scheduler` |
| **AI component** | Gemini (default `gemini-2.5-flash`, set with `GEMINI_MODEL`) accessed through the `google-genai` package |
| **Core AI feature** | Specialized model: a strict coach persona set by a rules prompt and few-shot examples |
| **Stretch features** | RAG over care notes, an agentic workflow with saved traces, a baseline vs specialized comparison, and a test harness |
| **Task** | Explain today's already-built care plan to a busy pet owner in under 120 words |
| **Intended users** | Pet owners using the PawPal+ app for everyday routines |
| **Out of scope** | Diagnosing illness, recommending medication or doses, replacing a vet, creating or changing the schedule |

The AI explains the plan. It never builds it: the deterministic `Scheduler` owns every time and conflict, and the coach is not allowed to mention a time that is not in the plan.

## 2. How the system works

1. The `Scheduler` produces today's plan and any overlapping requests.
2. The agent (`agent.py`) inspects the plan, retrieves care notes (`retriever.py`), and picks three few-shot examples (`data/examples.json`) that share the most tags with the day.
3. Gemini drafts a reply in a fixed format: `Top priority`, an optional `Heads-up`, and a `Tip` that cites a care note.
4. Ten guardrail checks run on the draft. A failing draft is revised once with the broken rules named, and if it still fails the built-in fallback explanation is shown instead.
5. Every step is saved to `ai_interactions.md` and logged to `coach.log`.

## 3. Specialization: baseline vs specialized

The specialization is behavioral (a system prompt plus few-shot examples), not weight fine-tuning. `compare_models.py` runs the same eight plans through three versions and scores each reply with the same ten checks.

| Version | Prompt |
|---|---|
| Baseline | "You are a helpful assistant." plus "Explain this pet care schedule." |
| Specialized | Persona, format, rules and three matching few-shot examples |
| Specialized + RAG | Everything above plus retrieved care notes and a citation rule |

> **TODO (replace before submitting):** run `python compare_models.py --out results/comparison.md` with your `GEMINI_API_KEY` set and paste the **Summary** table here. Do not paste `--dry-run` output, because that uses a stub, not Gemini.

> **TODO (replace before submitting):** write 2-3 sentences that name the measurable difference, for example "The baseline passed X of 80 checks and averaged N words; the specialized version passed Y of 80 and averaged M words." Use your real numbers.

### Before/after example

> **TODO (replace before submitting):** paste the "Before/after sample: Medication moved by a walk" block from `results/comparison.md`, showing the baseline, specialized and specialized+RAG replies, then add one or two sentences on what changed (format, length, citation, vet mention).

### RAG impact

Without RAG, the Tip is whatever the model chooses to say. With RAG, the Tip has to be based on a retrieved note and end with its source (for example `[dog.md]`), and a citation to a document the model was not shown is rejected by the `citation` check.

> **TODO (replace before submitting):** from your real comparison, state in one or two sentences how the specialized+RAG replies differ from the specialized-only replies, for example whether the `citation` check passed more often and whether the Tips became more specific.

## 4. Reliability and testing results

**Offline, no API key needed** (these are real results from the repo):

| Test | Result |
|---|---|
| New unit tests (`tests/test_coach.py`, `test_agent.py`, `test_retriever.py`) | 43 passed |
| Guardrail suite (`python evaluate.py --offline`), 12 scripted failure cases | 12 of 12 expectations met |

The guardrail suite feeds the real agent a scripted model that misbehaves on purpose: dosage language, invented times, invented sources, a missing citation, a missing vet mention, markdown, an over-long reply, a wrong format, a model that never complies, a model outage, and an empty plan. Each was caught, then either accepted after the retry or replaced by the fallback.

> **TODO (replace before submitting):** run `python evaluate.py --out results/evaluation.md` with your key and add the live result here, for example "Gemini passed all 10 checks in X of 8 scenarios, Y of them on the first try; Z used the fallback."

## 5. Limitations and biases

- **Rule checks cannot judge quality.** The guardrails verify format, length, banned phrases, grounded times and citations. They cannot tell whether a Tip is good advice, only that it follows the rules.
- **The care notes are tiny and written by me.** Seven short documents of general guidance, not veterinary sources. They cover dogs, cats, rabbits, fish and turtles only, and a bird, reptile or small mammal other than a rabbit gets no species note.
- **Species bias.** The notes and examples lean toward dogs and cats, so the coach is more useful for them. A species the retriever does not recognize gets only the general and medication notes, or none.
- **Keyword retrieval misses synonyms.** A "bath" task will not find the grooming note, and a task described in another language will not match anything.
- **Banned-phrase rules are blunt.** The dosage check looks for patterns such as "5 mg", so unusual wording could slip through, and a harmless phrase containing a blocked word could trigger a needless retry.
- **The few-shot examples are synthetic.** I wrote them myself, so the model learns my idea of a good reply rather than real owners' needs.
- **English and 24-hour times only.** The scheduler and prompts assume English text and `HH:MM` times.
- **It depends on an outside service.** If Gemini is down, over quota, or the key is missing, the owner gets the less personal fallback text.
- **Limited testing.** Eight scenarios and a scripted model do not cover every household, such as many pets, very long task names, or unusual species.

## 6. Potential for misuse and how it is prevented

| Risk | Prevention in this project | What is still open |
|---|---|---|
| The coach gives medical advice or a dose, and an owner trusts it | The prompt forbids diagnoses and amounts, the `no medical advice` check blocks them, and medication days must mention the vet | A cleverly worded unsafe reply might pass the pattern checks |
| The coach changes a medication time | It can only explain the plan, and the `grounded times` check rejects any time not in the plan | None for times; the scheduler stays deterministic |
| The coach invents facts or sources | Tips must be based on retrieved notes, and citations are checked against the notes shown | A note could be wrong, because I wrote them |
| Owners over-trust the AI | The app labels whether an answer came from Gemini or the fallback, shows the notes used, and exposes the reasoning steps | There is no on-screen "not veterinary advice" banner yet |
| Prompt injection through a task name (for example "ignore your rules") | The output checks still run on the reply, and times, format and citations are enforced in code | Task text is placed in the prompt without sanitizing |
| API key leaks | Key is read from an environment variable or `.streamlit/secrets.toml`, which is git-ignored | The user must keep the key out of commits |
| Private data sent to a third party | Only pet names, species, task names and times are sent | Owners should know the plan goes to Gemini |

## 7. What surprised me while testing reliability

> **TODO (write this yourself, from your own live runs):** describe one to three things you did not expect. Prompts to help you:
> - Which of the ten checks did Gemini fail most often on the first try?
> - Did the retry fix it, or did the fallback end up answering?
> - Did the baseline break rules you expected the specialized version to break, or the other way around?
> - Did anything the guardrail rejected turn out to be a fair reply (a false alarm)?
>
> Use real observations only. If your runs went smoothly, say that and say what you checked to be sure.
>
> Something real from development you can mention if it matches your experience: the first few-shot example test passed one example for the wrong reason (a fragile string split), and retrieval initially ranked all matches together, so walk notes pushed the medication note out of the top three until it was changed to pick the best note per task.

## 8. AI collaboration

I built this project with Claude as a coding and planning assistant, and the system itself uses Gemini at run time.

**Where AI helped.** Claude drafted the structure of the coach, the retriever, the agent, the evaluation harness and the architecture diagram, and it ran the tests offline to check its own work. I directed the scope: I chose the specialized model as my core feature, asked for all four stretch features, asked for output guardrails, and asked for each file to be delivered separately.

> **TODO (write this yourself):** one suggestion you **accepted**, and why it helped. Pick one that is true for you.
>
> Candidates from this project (use one only if it matches what you actually did and thought):
> - Splitting the work so the `Scheduler` owns times and the AI only explains them.
> - Having the guardrail retry once and then fall back, so the app never shows an unchecked reply.

> **TODO (write this yourself):** one suggestion you **rejected or changed**, and why. Pick one that is true for you.
>
> Candidates from this project (use one only if it matches what you actually did and thought):
> - You questioned why RAG was added when your core feature was specialization. The answer was that RAG was a required stretch, and you chose whether to keep it.
> - You asked for the files to be delivered separately instead of as a zip.
>
> Say what you decided, what you did instead, and what you learned from it.

> **TODO (write this yourself, optional):** one example where the AI was wrong or needed checking, and how you caught it.

## 9. Future improvements

- Replace the hand-written care notes with vetted sources and add more species.
- Use embeddings for retrieval so synonyms and rewordings match.
- Add a second check, such as a model-based judge, for whether a Tip is sensible, not just well formatted.
- Add an on-screen "general guidance, not veterinary advice" notice next to every answer.
- Sanitize task text before it goes into the prompt, and test prompt-injection cases.
- Cache answers per plan so repeated clicks do not repeat API calls.
- Test with real owners and revise the examples from their feedback.
- Add a measured latency and cost report to the comparison.
