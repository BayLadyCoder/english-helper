# Output quality log

Scores are 1 (poor), 2 (usable), or 3 (accurate and useful). Notes are from live `llama3.2` lookups after the prompt asked for English fields and a target-language translation.

| Lookup | Language | Definition (1-3) | Examples (1-3) | Notes |
|---|---|---|---|---|
| bicycle | Spanish | 3 | 2 | Definition and "bicicleta" are right. One example is a fragment. |
| bicycle | French | 3 | 3 | "Vélo" and the example sentences fit. |
| run | Spanish | 1 | 2 | Definition said "operate a vehicle." Examples still showed physical running. Translation "correr" is fine. |
| break the ice | Spanish | 3 | 3 | Idiom meaning is correct. Translation "romper a hielo" drops the article. |
| ephemeral | Spanish | 3 | 2 | Definition is right. One synonym came back in Spanish. |
| however | German | 1 | 2 | Defined as "in addition," which is wrong. Examples are grammatical but not a clear contrast. |
| bank | Spanish | 3 | 2 | Financial sense is correct. The model wrapped some examples in terminal color codes; the parser strips those. |
| under the weather | French | 3 | 1 | Definition is the idiom. One example treats it as actual weather. |
| She hit the books before the exam. | Spanish | 3 | 2 | Idiom is defined as studying. Examples support that. |
| casa | English | 2 | 2 | Defined as "house." Examples mix Spanish and English. Non-English input is less reliable. |
| serendipity | Japanese | — | — | Both attempts failed validation because `translation` was empty. The app shows a retry message and does not save a card. |

Section 13 cases covered here: common noun (`bicycle`), multiple meanings (`run`, `bank`), idiom (`break the ice`), and non-English input (`casa`). Empty input, overlong input, and a stopped model are checked in the UI and in `tests/test_llm_client.py` (`test_call_model_unavailable` points at a closed port and expects "Is Ollama running?").
