# Source of `congress_benchmark.jsonl`

This file is not covered by the [MIT license](../LICENSE) at the repository root.

`congress_benchmark.jsonl` is a JSONL conversion of the table *Table of Supreme Court Decisions Overruled by Subsequent Decisions*, compiled by the Congressional Research Service and published with the Constitution Annotated (Library of Congress):

- Table: <https://constitution.congress.gov/resources/decisions-overruled/>
- Methodology: [Appendix 1.2](https://constitution.congress.gov/browse/essay/appx.1/ALDE_00000065/)

A work of the United States Government, prepared by federal employees as part of their official duties, is not protected by copyright in the United States (17 U.S.C. § 105). This repository does not claim copyright in the CRS table and does not relicense it.

Changes made for this repository:

- The CRS CSV was converted to one JSONL object per overruling opinion, with the fields `id`, `case_citation`, and `citations` (`cite`, `class`).
- Decisions that the CSV identifies only by docket number were given their U.S. Reports citation so the pipeline can download the opinion.
- Parentheticals printed on an overruled citation, such as `(in part)`, are kept inside the `cite` string.
- Every citation's `class` is `overruled`, following the CRS table.

The same 236 pairs are the dataset in Li Zhang, Jaromir Savelka, and Kevin Ashley, [Do LLMs Truly Understand When a Precedent Is Overruled?](https://arxiv.org/abs/2510.20941) (arXiv:2510.20941, 2025). That paper records a prior use of the table. The source of the pairs remains the CRS compilation.
