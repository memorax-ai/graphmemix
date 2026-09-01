# H2HMem evidence-supported reporting track

## Scope

The official H2HMem release contains 2,236 questions and evaluates them after a
method has consumed every available session in a dialogue.  The official
baseline loaders also read questions from directories without `session.json`;
`session0` receives explicit query-image path handling.  Therefore this track
is a project reporting slice, not an official H2HMem subset.

Primary sources:

- paper: <https://arxiv.org/abs/2606.09461>
- code: <https://github.com/varib1/H2HMEM>
- dataset: <https://huggingface.co/datasets/varib/H2HMEM>
- locked data revision: `555a613df9dc462b42e4edd53ff97e799572da99`

## Locked selection

Protocol: `mmmb-h2hmem-evidence-supported-1.0`.

- official questions: 2,236
- selected questions: 1,982
- excluded: 254
  - 246 questions use `answer_session=["session0"]`; `session0` is a
    cross-session question container and has no `session.json` or memories.
  - 8 questions use `answer_session=["session6"]` in
    `dyadic/dialogue3`; that directory has questions and images but the locked
    official release has no `session.json`.
- recovered: 13 multi-party aliases such as `S3-2`, deterministically mapped
  to the matching dialogue's `session2`.

Generate the locked allowlist with:

```bash
python scripts/build_h2h_evidence_track.py
```

The resulting file is
`data/derived/h2hmem/evidence_supported_v1/question_ids.txt`.  H2H judge runs
must pass it through `--question-ids`.  Methods still ingest the complete
available dialogue context; only the reported question population changes.

The 2,236-question official-style full-history score may be retained as a
separately labelled auxiliary result.  It must not be compared with or merged
into the 1,982-question Main Track score.
