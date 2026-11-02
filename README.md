## 1. Project Overview

The objective of this project was to fine-tune a small Gemma language model to generate synthetic data that resembles the characteristics of its training datasets.

Rather than asking the model to answer questions or perform conventional text generation, I trained it to generate entirely new structured records using one fixed prompt:

```text
Generate synthetic data.
```

I implemented this experiment independently for two different domains:

1. **Insurance:** A combination of three insurance-related datasets containing customer questions, support responses, and insurance agent scenarios.
2. **Healthcare (MIMIC-IV):** Hospital admission records containing patient demographics, diagnoses, procedures, laboratory measurements, ICU information, and medications.

Both experiments used the same general approach: supervised fine-tuning (SFT) of a small Gemma model using Low-Rank Adaptation (LoRA).

The generated records were subsequently evaluated against the original datasets to assess their structural validity, statistical similarity, diversity, and potential issues.

### Project Objective

The primary objective was to determine whether a fine-tuned small language model could learn the underlying characteristics of structured datasets and reproduce those characteristics in newly generated records.

The goal was not simply to produce valid JSON. The generated records also needed to resemble the original data statistically and structurally.

**Important:** Synthetic similarity does not automatically mean factual accuracy, clinical validity, or guaranteed privacy. These are separate properties that require independent evaluation.

## 2. Project Architecture

The project consists of two independent fine-tuning pipelines.

```text
                    ORIGINAL DATASETS
                           |
               +-----------+-----------+
               |                       |
         INSURANCE DATA            MIMIC-IV DATA
               |                       |
         Data Loading             Data Loading
               |                       |
        Data Cleaning             Data Cleaning
               |                       |
       Schema Preparation        Admission-level
               |                 Record Construction
               |                       |
       Training Examples         Training Examples
               |                       |
               +-----------+-----------+
                           |
                    GEMMA BASE MODEL
                           |
                    LoRA FINE-TUNING
                           |
                 FINE-TUNED ADAPTERS
                           |
                  FIXED GENERATION PROMPT
                           |
                "Generate synthetic data."
                           |
                 SYNTHETIC JSON RECORDS
                           |
                JSON PARSING & VALIDATION
                           |
                STATISTICAL EVALUATION
                           |
                 CSV / JSON OUTPUT FILES
```

Each experiment has its own dataset, fine-tuned adapter, generation process, and evaluation pipeline.

## 3. Datasets

### 3.1 Insurance Experiment

The first experiment combines three insurance datasets.

| Dataset | Original Size | Records Used | Data Type |
|---|---:|---:|---|
| InsuranceQA V2 | 16,889 | 2,000 | Questions and answers |
| Bitext Insurance Chatbot | 39,000 | 2,000 | Structured support conversations |
| Insurance AI Reliability Benchmark | 510 | 1,020 | Insurance agent scenarios |
| **Total** | | **5,020** | |

#### A. InsuranceQA V2

Source: https://github.com/shuzi/insuranceQA/tree/master/V2

InsuranceQA contains real insurance-related customer questions and answers written by insurance professionals.

It provides examples of how insurance-related questions are formulated and how they are answered.

**Data characteristics:**
- Customer insurance questions.
- Professional answers.
- Insurance terminology.
- Different insurance-related topics.

I used 2,000 records from the available 16,889 records.

#### B. Bitext Insurance Chatbot Dataset

Source: https://huggingface.co/datasets/bitext/Bitext-insurance-llm-chatbot-training-dataset

This dataset contains insurance customer-support conversations and responses.

The dataset includes structured fields:

| Field | Description |
|---|---|
| instruction | Customer request or question |
| intent | Identified customer intent |
| category | Insurance-related category |
| tags | Associated labels or keywords |
| response | Generated support response |

**Dataset characteristics:**
- 39,000 original records.
- 39 different intents.
- 17 documented categories.
- 379 tags.

I used 2,000 records for fine-tuning.

#### C. Insurance AI Reliability Benchmark

Source: https://huggingface.co/datasets/pashas/insurance-ai-reliability-benchmark

This dataset contains 510 insurance-related agent scenarios.

Unlike conventional customer-support datasets, it focuses on insurance agent reliability and decision-making scenarios.

It includes situations where an agent should recognize when a case requires human intervention.

Since this dataset was relatively small compared with the other two datasets, I repeated its 510 records twice during training preparation.

This resulted in 1,020 training examples from this source.

**Note:** Repetition increases the number of training examples but does not introduce additional unique information.

### 3.2 Insurance Training Composition

The final insurance training dataset contained 5,020 examples.

| Source | Training Examples | Training Share |
|---|---:|---:|
| Bitext | 2,000 | 39.84% |
| InsuranceQA | 2,000 | 39.84% |
| Insurance Benchmark | 1,020 | 20.32% |
| **Total** | **5,020** | **100%** |

The model therefore learned from three different types of insurance information rather than a single homogeneous dataset.

This also introduced a challenge: the datasets had different schemas, field names, record lengths, and distributions.

### 3.3 Healthcare Experiment: MIMIC-IV Demo

Source: https://physionet.org/content/mimic-iv-demo/

The second experiment used the publicly available MIMIC-IV Clinical Database Demo.

MIMIC-IV is a clinical database containing hospital and ICU information.

For this experiment, I used the demo version rather than the full MIMIC-IV database.

**Dataset characteristics:**

- 275 hospital admissions.
- 100 patients.
- Multiple clinical tables.
- Structured patient and hospital information.

The original dataset contains multiple relational tables. I transformed the relevant information into one consolidated JSON record per hospital admission.

#### Information included in each record

| Attribute | Description |
|---|---|
| Demographics | Patient age and gender |
| Hospital Admission | Admission and discharge information |
| Length of Stay | Duration of hospital stay |
| ICU Information | ICU admission and stay duration |
| Diagnoses | Diagnoses associated with the admission |
| Procedures | Medical procedures |
| Laboratory Measurements | Selected laboratory results |
| Medications | Selected medication information |

Instead of training directly on individual database tables, I constructed admission-level JSON records.

This allowed the model to learn relationships between different clinical attributes within a hospital admission.

#### MIMIC-IV Data Preparation

The original 275 admissions were divided into training and held-out data using a patient-level split.

| Split | Admissions |
|---|---:|
| Training | 249 |
| Held-out | 26 |
| **Total** | **275** |

The split was performed by patient rather than randomly by individual admission.

This is important because a patient can have multiple hospital admissions. Keeping the held-out patients separate reduces the risk of patient-level information leakage between training and evaluation.

#### Training Example Expansion

Each of the 249 training admissions was represented eight times.

For each representation, a random subset of laboratory measurements and medications was selected.

This produced:

249 admissions × 8 representations = 1,992 training examples.

The purpose was to expose the model to different combinations of laboratory and medication information while retaining the underlying admission-level characteristics.

**Important distinction:** The 1,992 examples are not 1,992 independent hospital admissions. They are eight representations of each of the 249 original training admissions.

## 4. Model and Fine-Tuning Methodology

Both experiments used a small instruction-tuned Gemma model.

### Base Model

`unsloth/gemma-4-E2B-it`

The model was loaded using Unsloth to make fine-tuning more memory-efficient.

### Fine-Tuning Method: LoRA

I used Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique.

Instead of updating all the parameters of the original language model, LoRA introduces smaller trainable matrices into selected model layers.

The original model weights remain frozen while the adapter parameters are optimized.

**Configuration:**

| Parameter | Value |
|---|---|
| Base model | unsloth/gemma-4-E2B-it |
| Fine-tuning | Supervised Fine-Tuning (SFT) |
| Adaptation | LoRA |
| LoRA rank (r) | 16 |
| Trainable parameters | Approximately 0.49% |
| Learning rate | 0.0001 |
| Random seed | 3407 |
| Effective batch size | 8 |
| GPU | Tesla T4 |
| Environment | Google Colab |

### Why LoRA?

Full fine-tuning updates the complete model, which requires substantially more GPU memory and computational resources.

LoRA makes the experiment feasible on a free Colab Tesla T4 by training only a small fraction of the model parameters.

Benefits include:

- Lower memory requirements.
- Reduced training computation.
- Smaller adapter checkpoints.
- Preservation of the original model weights.
- Easier experimentation with separate datasets.

### Training Configuration

| Setting | Insurance | MIMIC-IV |
|---|---:|---:|
| Training examples | 5,020 | 1,992 |
| Training steps | 200 | 249 |
| Epochs | Partial epoch | 1 |
| Effective batch size | 8 | 8 |
| Learning rate | 1e-4 | 1e-4 |
| LoRA rank | 16 | 16 |

The insurance experiment used 200 optimization steps. At an effective batch size of eight, this corresponds to approximately 1,600 example presentations, assuming the configured batch size is maintained throughout training.

Therefore, the insurance model did not complete a full epoch.

The MIMIC-IV experiment completed one epoch over its 1,992 training examples, corresponding to approximately 249 optimization steps.

## 5. Training Data Formatting

The two experiments use a consistent instruction-based training approach.

Every training example contains:

1. A fixed instruction.
2. One structured JSON record.

The instruction is:

```text
Generate synthetic data.
```

An illustrative example of the training format is:

```text
Generate synthetic data.

{
  "age": 65,
  "gender": "F",
  "hospital_stay_days": 6,
  "icu_stay_days": 2,
  "diagnoses": [...],
  "procedures": [...],
  "lab_results": [...],
  "medications": [...]
}
```

The example above illustrates the format, not an actual record from the dataset.

### Response-Only Training

An important part of the training setup is that the loss is calculated only on the JSON response, not on the instruction.

This means the model is trained to generate the record after receiving the prompt rather than learning to reproduce the prompt itself.

The training objective is to predict the next tokens of the structured JSON output.

Conceptually:

```text
Input:
Generate synthetic data.

Target:
The complete JSON record
```

This is implemented using the response-only training functionality from Unsloth.

## 6. Synthetic Data Generation

After fine-tuning, each adapter is used to generate new records.

The generation process follows the same general approach for both domains.

### Generation Workflow

1. Load the base Gemma model.
2. Load the corresponding fine-tuned LoRA adapter.
3. Provide the fixed instruction.
4. Generate a new JSON response.
5. Extract and parse the JSON.
6. Validate the required fields.
7. Save valid records to CSV and JSON files.

### Generation Prompt

```text
Generate synthetic data.
```

The purpose of using one fixed prompt is to test whether the model has learned the dataset characteristics during fine-tuning.

The model is not given an individual real record as a generation example.

### Sampling Temperature

For MIMIC-IV, I tested two generation configurations.

| Configuration | Temperature | Generated Dataset |
|---|---:|---|
| Run A | 1.0 | Initial 10-record run |
| Run B | 0.7 | Lower-temperature 30-record run |

Temperature controls the randomness of token selection.

- Higher temperature generally increases output variation and randomness.
- Lower temperature generally produces more predictable outputs.

Run B produced better aggregate results for several numerical attributes, although a lower temperature does not guarantee better statistical similarity in every situation.

## 7. Evaluation Methodology

Generating JSON successfully does not necessarily mean that synthetic data resembles the original dataset.

Therefore, I evaluated the generated records using multiple criteria.

### 7.1 Structural Validity

Checks whether generated records:

- Can be parsed as JSON.
- Contain the expected fields.
- Follow the required record structure.

This measures formatting and schema compliance.

### 7.2 Duplicate Detection

Checks whether generated records are exact copies of training records or duplicates of other generated records.

For MIMIC-IV, the evaluation also checks against held-out admissions.

This is important because a synthetic dataset should not simply reproduce its training examples.

However, exact duplicate checks alone cannot establish privacy or rule out near-duplicates.

### 7.3 Distribution Similarity

Checks whether the generated data follows similar distributions to the original data.

For example:

- Distribution of patient ages.
- Hospital stay duration.
- Number of diagnoses.
- Medication counts.
- Insurance categories.
- Customer intent distribution.

### 7.4 Numerical Range Validation

Compares the minimum and maximum values of numerical attributes between the real and synthetic datasets.

For MIMIC-IV, laboratory measurements were checked against the observed ranges in the real data.

### 7.5 Categorical Distribution

Categorical attributes were evaluated using Total Variation Distance (TVD).

TVD measures the difference between two categorical probability distributions.

The range is:

```text
0 = Identical distributions

1 = Completely different distributions
```

A lower TVD indicates better distributional similarity.

For example, if the original dataset contains a particular category in 20% of records and the generated dataset contains it in approximately 20%, the distributions are relatively similar.

### 7.6 Record Characteristics

I also compared:

- Average record length.
- Source composition.
- Number of diagnoses and procedures.
- Number of medications.
- ICU admission proportions.

These checks help determine whether the model has learned more than just the JSON structure.

## 8. Results

All reported results below are taken from the experiment notebooks.

The sample sizes are small, so the results should be treated as preliminary experimental evidence rather than statistically conclusive findings.

### 8.1 Insurance Results

| Evaluation | Result |
|---|---:|
| Generated records | 312 |
| Records passing parsing and field checks | 305 |
| Records retained | 300 |
| Validity rate | 97.8% |
| Exact copies of training records among retained records | 0 |
| Duplicate generated records among retained records | 0 |

The model demonstrated strong structural generation performance.

It was able to generate records that could be parsed and retained according to the implemented field checks.

#### Source Distribution

The generated records were also compared against the original training-source proportions.

| Source | Real Training Mix | Synthetic Mix |
|---|---:|---:|
| Bitext | 39.8% | 38.7% |
| InsuranceQA | 39.8% | 43.0% |
| Insurance Benchmark | 20.3% | 18.3% |

The generated source distribution was reasonably close to the original training composition.

#### Record Length

The average generated record length was also compared with the corresponding real records.

For example, the Insurance Benchmark had an average record length of approximately:

- Real: 1,209 characters.
- Synthetic: 1,207 characters.

This suggests that the model learned some of the structural and length characteristics of the original records.

#### Categorical Similarity

Categorical distributions were less consistent.

Some fields demonstrated good similarity, while others differed substantially.

| Field | TVD | Interpretation |
|---|---:|---|
| Benchmark channel | 0.025 | Very close distribution |
| Bitext intent | 0.699 | Large distribution difference |

This indicates that the model learned certain categorical patterns better than others.

In particular, rare or highly varied labels were more difficult to reproduce accurately.

**Insurance conclusion:** The model performed well at producing structured records and approximating some overall dataset characteristics. However, categorical distribution matching remains an important limitation.

### 8.2 MIMIC-IV Results

Two generation runs were evaluated.

| Metric | Real Data (275) | Run A (10) | Run B (30) |
|---|---:|---:|---:|
| Mean age | 62.73 | 66.10 | 62.80 |
| Mean hospital stay (days) | 6.88 | 7.87 | 5.98 |
| Mean ICU stay (days) | 3.63 | 5.62 | 1.60 |
| Mean diagnoses per stay | 16.39 | 17.90 | 16.67 |
| Mean medications | 31.08 | 37.30 | 30.73 |
| Share with ICU stay | 0.47 | 0.40 | 0.47 |

#### Run A: Temperature 1.0

The initial run generated 10 records.

It showed that the model could produce structured clinical records, but some numerical characteristics differed from the original dataset.

For example:

- Mean age was higher.
- Mean hospital stay was higher.
- Mean ICU stay was substantially higher.
- Medication count was higher.

#### Run B: Temperature 0.7

The second run generated 30 records using a lower sampling temperature.

This run showed better agreement for several aggregate metrics.

Examples:

- Mean age: 62.80 versus 62.73.
- Mean diagnoses: 16.67 versus 16.39.
- Mean medications: 30.73 versus 31.08.
- ICU admission share: 47% versus 47%.

The mean hospital stay was 5.98 days compared with 6.88 days in the real dataset.

The mean ICU stay was 1.60 days compared with 3.63 days in the real dataset.

Therefore, Run B improved some metrics but did not consistently match all numerical characteristics.

#### Additional Run B Checks

| Evaluation | Result |
|---|---:|
| Generated records passing validation | 32 of 32 |
| Records retained in the reported sample | 30 |
| Generated laboratory values | 238 |
| Laboratory values within observed real-data min-max ranges | 238 |
| Exact copies of training or held-out admissions | 0 |

All 238 generated laboratory values fell within the observed minimum and maximum ranges of the corresponding real laboratory measurements.

This demonstrates range compliance for the evaluated values, but does not establish that the values are clinically plausible or that their distributions match the real measurements.

**MIMIC-IV conclusion:** The model learned several admission-level numerical patterns and generated structurally valid records. However, medical coding, medication naming, and certain demographic distributions remain problematic.

## 9. Limitations and Observed Issues

It is important to distinguish between successful JSON generation and genuinely reliable synthetic data.

The experiments revealed several limitations.

### 9.1 Limited Evaluation Samples

The evaluation used:

- 300 retained insurance records.
- 30 retained MIMIC-IV records.

These sample sizes are insufficient for strong conclusions about rare categories, tail distributions, or overall statistical fidelity.

The results are preliminary.

### 9.2 Incorrect Medical Codes

One of the most significant problems was medical coding accuracy.

Only 18.8% of generated diagnosis codes matched their associated titles in the real ICD dictionary.

For procedure codes, the observed match rate was 0%.

Additionally, approximately half of the records contained inconsistencies between the number of diagnoses or procedures listed and their corresponding declared counts.

This indicates that the model learned some of the general structure of clinical records without reliably learning the relationships between medical codes and their official descriptions.

### 9.3 Demographic Distribution Differences

The generated MIMIC-IV sample did not fully reproduce the original demographic distribution.

For example:

| Demographic Attribute | Real Data | Synthetic Data |
|---|---:|---:|
| White patients | 62% | 87% |

No deaths were generated, while the original observed death rate was 5.5%.

Although zero deaths in a sample of 30 can occur by chance, the demographic difference is a more substantial distribution mismatch.

These results show that learning general record patterns does not guarantee accurate reproduction of every demographic or outcome distribution.

### 9.4 Medication Naming Issues

Some generated medication names were malformed or appeared to be invented.

One observed example was:

```text
Glycombinant Adrenalin (Flagyl)
```

Medication names were not systematically validated against an authoritative medication database.

Consequently, the generated medication information should not be considered medically reliable.

### 9.5 Insurance Dataset Noise

The insurance datasets also contained some inconsistencies.

Examples included:

- An InsuranceQA record with unexpected metadata fields.
- A benchmark intent containing a typo: `file_clacked_claim`.
- Multiple near-duplicate category labels in Bitext, including `PAYMENT`, `PAYMENTS`, and `PAY`.

These inconsistencies may have influenced the model's ability to reproduce categorical distributions.

### 9.6 Insurance Response Accuracy

The insurance experiment evaluated structural and statistical characteristics.

It did not comprehensively verify whether the generated insurance responses were factually correct, legally accurate, or consistent with actual policy conditions.

Therefore, valid JSON should not be interpreted as a correct insurance answer.

### 9.7 Training Duration

The insurance model was trained for 200 optimization steps.

At an effective batch size of eight, this represents approximately 1,600 example presentations out of 5,020 available training examples.

It therefore received approximately one-third of an epoch's worth of training examples.

The MIMIC-IV model completed one epoch.

Longer training could potentially improve learning, but additional epochs would need to be evaluated carefully for overfitting and memorization.

### 9.8 Privacy and Re-identification

Synthetic data should not automatically be considered anonymous or privacy-preserving.

The experiments included exact duplicate checks, but these checks do not measure membership inference risk, attribute disclosure, or near-duplicate memorization.

A dedicated privacy evaluation would be necessary before using generated clinical records for sensitive applications.

## 10. Key Findings

The experiments demonstrate that a small instruction-tuned language model can learn aspects of structured datasets through parameter-efficient fine-tuning.

The insurance experiment demonstrated:

- Strong JSON parsing and field compliance.
- No exact training-record copies among the 300 retained records.
- Reasonably close source-level proportions.
- Good similarity for some categorical fields.
- Significant differences in other categorical distributions.

The MIMIC-IV experiment demonstrated:

- Successful generation of admission-level clinical records.
- Close agreement for several aggregate numerical metrics.
- All evaluated laboratory values within the corresponding observed real-data ranges.
- Structural validation of all 32 records in the reported second-run validation.
- Significant limitations in medical coding and medication naming.

The most important finding is that **learning a dataset's structure and numerical ranges is not the same as learning its complete statistical or domain-specific meaning.**

A language model can generate records that look structurally convincing while still producing incorrect codes, invalid terminology, or unrealistic combinations of attributes.

This makes independent validation an essential component of any synthetic data generation pipeline.
