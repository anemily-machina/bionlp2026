# MedDec: A Dataset for Extracting Medical Decisions from Discharge Summaries

## Dataset files
  - data/: contains the json files of the dataset.
  - README.md: this file.

## Stats
  - 419 patients
  - 451 notes 

## Splits
  - train: 350 notes
  - val: 53 notes
  - test: 48 notes

### Important Note About Test Set
  - The test set is not released with the dataset. 
  - The test set will be released publicly after the shared task is over.
  - Details about the shared task will be announced soon.

## Definition of IDs in the dataset:
  - SUBJECT_ID: a unique patient
  - HADM_ID: a unique admission to the hospital
  - ROW_ID: a row identifier unique a table in MIMIC.

  The IDs are derived from the MIMIC-III dataset, and they can be used to join this dataset with other tables in MIMIC-III.

## Format of json file names:
  - [SUBJECT_ID]_[HADM_ID]_[ROW_ID].json

## Fields in the json files:
  - annotator_id: ID of annotator.
  - discharge_summary_id: Unique identifier for each discharge summary.
  - annotations: List of annotated spans, each containing:
    - start_offset: The starting character index of the span.
    - end_offset: The ending character index of the span.
    - category: The category of the medical decision.
    - decision: The text of the annotated span.
    - annotation_id: Unique identifier for each annotation.

## Project Home Page: 
  https://github.com/CLU-UML/MedDec
