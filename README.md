# Overruling retrieval

This project reads U.S. judicial opinions and records how each opinion treats the cases it cites. An LLM assigns those treatments from a chosen taxonomy, in which overruling sits alongside other negative treatment, procedural signals, and finer distinctions. It is meant to measure that extraction on labeled benchmarks and to run it across reporter volumes where no ground truth exists.

**Evaluate** loads a JSONL benchmark, downloads each opinion, sends it to an LLM, and scores the answer against the benchmark.

**Production** runs the same extraction over a reporter volume from the [Caselaw Access Project](https://static.case.law/) and stores the relations in SQLite, without ground truth.

Run the commands from the repository root.

## Quick start

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

Set one LLM key in `.env`. The examples use OpenAI (`OPENAI_API_KEY`). Model names are in [`config/models.yaml`](config/models.yaml).

Opinions are read from a local cache, then from CourtListener if `COURTLISTENER_API_TOKEN` is set, then from the Caselaw Access Project static files. A CourtListener token is strongly recommended. Without it, opinions that are missing from the static files cannot be retrieved, so not every benchmark entry can be run. That includes Congress entries 1–14 (U.S. Reports after volume 572) and a few OWN entries in the Federal Reporter Fourth series. Create a free token as described in the [CourtListener REST API docs](https://www.courtlistener.com/help/api/rest/) and put it in `.env`. The token itself is on your [CourtListener API profile](https://www.courtlistener.com/profile/api/).

This writes the prompt for one opinion and does not call a model:

```bash
python evaluate.py \
  --path_to_dataset benchmarks/congress_benchmark.jsonl \
  --api_connector jsonl \
  --prompt_style a \
  --indices 15 \
  --batch
```

The same case, scored against the benchmark:

```bash
python evaluate.py \
  --path_to_dataset benchmarks/congress_benchmark.jsonl \
  --api_connector openai \
  --llm_model gpt-4.1-mini \
  --prompt_style a \
  --indices 15
```

Results are written to `output/evaluation_results_<timestamp>/`.

## Example runs

Leave out `--indices`, `--from_idx`, and `--to_idx` to run the whole file. `--api_connector` accepts `openai`, `claude`, `gemini`, `deepseek`, and `ollama`.

Congress benchmark, short overruling prompt:

```bash
python evaluate.py \
  --path_to_dataset benchmarks/congress_benchmark.jsonl \
  --api_connector openai \
  --llm_model gpt-4.1-mini \
  --prompt_style a \
  --from_idx 15 --to_idx 30
```

OWN04, four-label prompt. The file contains opinions with and without an overruling.

```bash
python evaluate.py \
  --path_to_dataset benchmarks/own_benchmark-OWN04.jsonl \
  --api_connector openai \
  --llm_model gpt-4.1-mini \
  --prompt_style j \
  --indices 1
```

`--prompt_style a` is the overruling prompt for the Congress benchmark. `j` is the four-label prompt for OWN01, OWN03, and OWN04. `i` is the fine-grained prompt for OWN02. `python evaluate.py --help` lists the others.

## Benchmarks

One treating opinion per line:

```json
{"id": 1, "case_citation": "Childers v. Floyd, 642 F.3d 953 (2011)", "citations": [{"cite": "Espy v. Massac, 443 F.3d 1362 (2006)", "class": "overruled"}]}
```

An empty `citations` list means that opinion has no ground-truth relation for the labels in that file.

| File | Entries | Contents | Prompt |
| --- | --- | --- | --- |
| `benchmarks/congress_benchmark.jsonl` | 236 | Supreme Court overruling pairs (292 `overruled` citations). | `a` |
| `benchmarks/own_benchmark-OWN01.jsonl` | 32 | KeyCite overruling only (64 `overruled` citations). | `j` |
| `benchmarks/own_benchmark-OWN02.jsonl` | 73 | Those overrulings plus other red-flag treatments, kept as the KeyCite verbs. | `i` |
| `benchmarks/own_benchmark-OWN03.jsonl` | 100 | All collected opinions, with the labels `overruled`, `overruling_recognized`, `bad_law`, and `negative_treatment`. | `j` |
| `benchmarks/own_benchmark-OWN04.jsonl` | 100 | Same opinions as OWN03. Only `overruled` citations are listed, so 67 opinions have an empty list. | `j` |

OWN01–OWN04 were compiled using Westlaw KeyCite. They share one set of opinions and differ in which entries and labels they include.

`benchmarks/congress_benchmark.jsonl` is a JSONL conversion of the Congressional Research Service table [Table of Supreme Court Decisions Overruled by Subsequent Decisions](https://constitution.congress.gov/resources/decisions-overruled/). The provenance note is [`benchmarks/congress_benchmark.SOURCE.md`](benchmarks/congress_benchmark.SOURCE.md).

## Production

Downloads one reporter volume and extracts treatments for each case in a page range. Relations are appended to `output/citation_relations.sqlite`.

```bash
python production.py \
  --reporter "U.S." \
  --volume 542 \
  --min_page 1 --max_page 20 \
  --api_connector openai \
  --llm_model gpt-4.1-mini \
  --prompt_style j
```

Leave out `--min_page` and `--max_page` to run the whole volume. `--volume` also accepts a list (`540,542`) or a range (`540-545`). Page limits apply only to a single volume.
