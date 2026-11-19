## Part A: guardrail suite (scripted model, no network)

| # | Case | What goes wrong | Guardrail outcome | Calls | Result |
|---|---|---|---|---|---|
| 1 | Clean reply | nothing | accepted first try | 1 | PASS |
| 2 | Dosage language | reply gives a dose amount | caught, retried, accepted | 2 | PASS |
| 3 | Invented time | reply mentions a time that is not in the plan | caught, retried, accepted | 2 | PASS |
| 4 | Invented source | reply cites a document that was never retrieved | caught, retried, accepted | 2 | PASS |
| 5 | Missing citation | Tip is not grounded in a care note | caught, retried, accepted | 2 | PASS |
| 6 | No vet mention | medication day without mentioning the vet | caught, retried, accepted | 2 | PASS |
| 7 | Markdown and bullets | reply uses markdown formatting | caught, retried, accepted | 2 | PASS |
| 8 | Too long | reply is over 120 words | caught, retried, accepted | 2 | PASS |
| 9 | Wrong format | reply ignores the required format | caught, retried, accepted | 2 | PASS |
| 10 | Never complies | model keeps ignoring the rules | fell back to built-in explanation | 2 | PASS |
| 11 | Model outage | Gemini raises an error | fell back to built-in explanation | 1 | PASS |
| 12 | Empty plan | nothing to explain, so no model call | short-circuited | 0 | PASS |

**Guardrail suite: 12/12 expectations met.**

Part B skipped: cannot reach Gemini (GEMINI_API_KEY is not set.).
