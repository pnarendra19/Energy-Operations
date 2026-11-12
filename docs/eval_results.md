# Evaluation results

Generator: **extractive (offline)** | top_k=4 | alpha=0.5 | chunk<= 80 words

Dataset: 8 SOPs, 42 chunks, 30 questions (24 in-scope, 6 out-of-scope)

| Metric | Value |
|---|---:|
| retrieval: recall@1 | 1.0 |
| retrieval: recall@3 | 1.0 |
| retrieval: recall@4 | 1.0 |
| retrieval: mrr | 1.0 |
| retrieval: fact_recall@4 | 1.0 |
| answer: fact_recall | 0.979 |
| answer: exact_answer_rate | 0.958 |
| answer: cites_only_correct_sop_rate | 1.0 |
| safety: abstention_on_out_of_scope | 1.0 |
| safety: false_abstain_rate | 0.0 |
| safety: groundedness_rate | 1.0 |
| safety: citation_validity_rate | 1.0 |
| hallucination_checker: answers_tested | 24 |
| hallucination_checker: fabricated_sentence_detected | 1.0 |
| hallucination_checker: altered_number_detected | 1.0 |
| hallucination_checker: missing_citations_detected | 1.0 |

## Retriever ablation

| Retriever | Title in embedding | Recall@1 | MRR | Fact-recall@k |
|---|---|---:|---:|---:|
| sparse only (TF-IDF) | True | 0.958 | 0.979 | 1.0 |
| dense only (LSA) | True | 0.958 | 0.979 | 1.0 |
| hybrid a=0.5 | True | 0.958 | 0.979 | 1.0 |
| hybrid + doc prior 0.25 | True | 0.958 | 0.979 | 1.0 |
| hybrid + doc prior + key-term penalty (default) | True | 1.0 | 1.0 | 1.0 |
| sparse only (TF-IDF) | False | 0.917 | 0.958 | 0.958 |
| dense only (LSA) | False | 0.958 | 0.979 | 0.958 |
| hybrid a=0.5 | False | 0.958 | 0.979 | 0.958 |
| hybrid + doc prior 0.25 | False | 0.958 | 0.979 | 1.0 |
| hybrid + doc prior + key-term penalty (default) | False | 0.958 | 0.979 | 1.0 |

## Per-question results

| ID | In scope | Abstained | Top doc | Score | Cover | Verdict | Fact recall | Correct SOP cited |
|---|---|---|---|---:|---:|---|---:|---|
| Q01 | True | False | SOP-SAF-001 | 0.385 | 1.0 | grounded | 1.0 | True |
| Q02 | True | False | SOP-SAF-001 | 0.277 | 0.6 | grounded | 1.0 | True |
| Q03 | True | False | SOP-SAF-002 | 0.425 | 0.833 | grounded | 1.0 | True |
| Q04 | True | False | SOP-SAF-002 | 0.426 | 0.833 | grounded | 1.0 | True |
| Q05 | True | False | SOP-SAF-002 | 0.521 | 1.0 | grounded | 1.0 | True |
| Q06 | True | False | SOP-MNT-101 | 0.345 | 1.0 | grounded | 1.0 | True |
| Q07 | True | False | SOP-MNT-101 | 0.48 | 0.833 | grounded | 1.0 | True |
| Q08 | True | False | SOP-MNT-101 | 0.264 | 0.667 | grounded | 1.0 | True |
| Q09 | True | False | SOP-MNT-102 | 0.324 | 0.75 | grounded | 1.0 | True |
| Q10 | True | False | SOP-MNT-102 | 0.576 | 1.0 | grounded | 1.0 | True |
| Q11 | True | False | SOP-MNT-103 | 0.51 | 0.833 | grounded | 1.0 | True |
| Q12 | True | False | SOP-MNT-103 | 0.402 | 0.75 | grounded | 1.0 | True |
| Q13 | True | False | SOP-MNT-103 | 0.517 | 1.0 | grounded | 1.0 | True |
| Q14 | True | False | SOP-OPS-201 | 0.526 | 1.0 | grounded | 1.0 | True |
| Q15 | True | False | SOP-OPS-201 | 0.639 | 1.0 | grounded | 1.0 | True |
| Q16 | True | False | SOP-EMR-301 | 0.522 | 0.875 | grounded | 1.0 | True |
| Q17 | True | False | SOP-EMR-301 | 0.444 | 0.8 | grounded | 1.0 | True |
| Q18 | True | False | SOP-OPS-202 | 0.564 | 1.0 | grounded | 1.0 | True |
| Q19 | True | False | SOP-OPS-202 | 0.43 | 0.8 | grounded | 1.0 | True |
| Q20 | True | False | SOP-OPS-202 | 0.603 | 1.0 | grounded | 1.0 | True |
| Q21 | True | False | SOP-MNT-102 | 0.428 | 1.0 | grounded | 1.0 | True |
| Q22 | True | False | SOP-SAF-001 | 0.561 | 1.0 | grounded | 1.0 | True |
| Q23 | True | False | SOP-OPS-201 | 0.518 | 1.0 | grounded | 0.5 | True |
| Q24 | True | False | SOP-SAF-002 | 0.431 | 1.0 | grounded | 1.0 | True |
| U01 | False | True | SOP-EMR-301 | 0.0 | 0.0 | abstained | - | - |
| U02 | False | True | SOP-MNT-103 | 0.116 | 0.5 | abstained | - | - |
| U03 | False | True | SOP-MNT-102 | 0.083 | 0.5 | abstained | - | - |
| U04 | False | True | SOP-SAF-002 | 0.05 | 0.167 | abstained | - | - |
| U05 | False | True | SOP-EMR-301 | 0.058 | 0.4 | abstained | - | - |
| U06 | False | True | SOP-SAF-002 | 0.2 | 0.429 | abstained | - | - |
