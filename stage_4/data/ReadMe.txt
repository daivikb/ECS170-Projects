Stage 4 uses the instructor's stage_4_data.zip.

For training, extract its contents so this folder contains:
  text_classification/train/neg/*.txt (12500 files)
  text_classification/train/pos/*.txt (12500 files)
  text_classification/test/neg/*.txt  (12500 files)
  text_classification/test/pos/*.txt  (12500 files)
  text_generation/data              (CSV with ID,Joke columns; 1622 rows)

Do not rename the extensionless generation file. Ignore __MACOSX and .DS_Store.
The complete working folder contains the original data. The submission ZIP excludes
raw datasets; its saved models contain vocabularies and learning histories and can
be used for generation and plotting without the datasets.

Movie review labels: neg=0, pos=1. Numeric ratings in filenames are not model inputs.
The 25000 supplied training reviews are split into 22500 training and 2500 validation
reviews with stratification and random_state=170. The supplied test set is untouched.

Jokes are normalized; 15 duplicate rows and two entries shorter than three words
are excluded. The remaining 1605 unique jokes are then split 90/10
with random_state=170. This yields 1444 training jokes and 161 validation jokes.
The generation data has no instructor-provided test partition. Validation perplexity
is used for checkpoint selection and is not an independent test estimate.
